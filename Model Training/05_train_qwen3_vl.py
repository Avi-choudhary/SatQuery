import os
import sys
import json
import argparse
import time
from PIL import Image
from tqdm import tqdm

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

class BigEarthNetVLMDataset(Dataset):
    def __init__(self, json_path, processor, max_samples=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if max_samples:
            self.data = self.data[:max_samples]
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        messages = item["messages"]
        
        # Load satellite image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Fallback if image fails to open
            image = Image.new("RGB", (120, 120), color=(0, 0, 0))

        # Format full conversation (user prompt + assistant response)
        full_text = self.processor.apply_chat_template(messages, tokenize=False)
        
        # Format user prompt only to find where the assistant response starts
        user_only_messages = [messages[0]]
        prompt_text = self.processor.apply_chat_template(user_only_messages, tokenize=False, add_generation_prompt=True)
        
        # Process inputs
        inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt"
        )
        
        # Prompt token length for masking
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt"
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # Build labels: mask prompt tokens with -100 so loss is only calculated on assistant answers
        labels = input_ids.clone()
        labels[:min(prompt_len, len(labels))] = -100

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        if "mm_token_type_ids" in inputs:
            res["mm_token_type_ids"] = inputs["mm_token_type_ids"].squeeze(0)
        if "pixel_values" in inputs:
            res["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in inputs:
            res["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
            
        return res

def collate_fn(batch):
    # Dynamic padding for batching
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    pad_id = 151643  # Standard pad token id for Qwen
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    batch_dict = {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels,
    }
    
    if "mm_token_type_ids" in batch[0]:
        batch_dict["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
            [item["mm_token_type_ids"] for item in batch], batch_first=True, padding_value=0
        )
    if "pixel_values" in batch[0]:
        batch_dict["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)
    if "image_grid_thw" in batch[0]:
        batch_dict["image_grid_thw"] = torch.stack([item["image_grid_thw"] for item in batch], dim=0)
        
    return batch_dict

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print("[SatQuery] Qwen3-VL-2B BigEarthNet LoRA Fine-Tuning Pipeline")
    print("="*60)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Target Model: {args.model_id}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Batch Size: {args.batch_size} (Grad Accum: {args.grad_accum}, Effective: {args.batch_size * args.grad_accum})")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Dry Run Mode: {args.dry_run}")
    print("="*60 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    print("1. Loading Processor...")
    processor = AutoProcessor.from_pretrained(args.model_id)

    print("2. Loading Model in bfloat16...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Freeze Vision Encoder to preserve optical satellite representations & save VRAM
    if hasattr(model, "visual"):
        print("Freezing Vision Encoder...")
        model.visual.requires_grad_(False)

    print("3. Applying LoRA Adapter...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("4. Preparing Datasets...")
    max_train = 20 if args.dry_run else args.max_samples
    train_dataset = BigEarthNetVLMDataset(args.train_file, processor, max_samples=max_train)
    val_dataset = BigEarthNetVLMDataset(args.val_file, processor, max_samples=10 if args.dry_run else 300)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )

    total_training_steps = (len(train_loader) // args.grad_accum) * args.epochs
    if total_training_steps == 0:
        total_training_steps = 10
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_training_steps * 0.05),
        num_training_steps=total_training_steps
    )

    print("\n5. Starting Training...")
    global_step = 0
    model.train()

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            # Move batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum

            loss.backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                vram_used = torch.cuda.memory_allocated(device) / (1024 ** 3)
                pbar.set_postfix({
                    "loss": f"{loss.item() * args.grad_accum:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "vram": f"{vram_used:.1f}GB"
                })

            if args.dry_run and step >= 10:
                print("\n[Dry Run] Completed 10 verification steps successfully!")
                break

        avg_train_loss = epoch_loss / max(1, step + 1)
        print(f"\nEpoch {epoch+1} Complete. Average Training Loss: {avg_train_loss:.4f}")

        # Quick validation
        print("Evaluating on validation split...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    v_outputs = model(**val_batch)
                    val_loss += v_outputs.loss.item()
        
        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Validation Loss: {avg_val_loss:.4f}")
        model.train()

        # Save Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"[BEST] New best validation loss ({best_val_loss:.4f})! Saving checkpoint...")
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)

        if args.dry_run:
            break

    # Save final model
    print(f"\nSaving final LoRA adapter to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("\n" + "="*60)
    print("[SUCCESS] Training successfully completed!")
    print(f"LoRA weights and processor saved in: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL on BigEarthNet with LoRA")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--train_file", type=str, default="data/qwen_train.json")
    parser.add_argument("--val_file", type=str, default="data/qwen_val.json")
    parser.add_argument("--output_dir", type=str, default="output/qwen3_vl_satquery_lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="Run 10 steps to test pipeline and memory")

    args = parser.parse_args()
    train(args)
