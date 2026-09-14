import os
import sys
import json
import argparse
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

def test_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print("[SatQuery] Qwen3-VL-2B Satellite Inference Test")
    print("="*60)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Base Model: {args.base_model}")
    print(f"LoRA Adapter: {args.lora_dir if os.path.exists(args.lora_dir) else 'None (Running Base Zero-Shot)'}")
    print("="*60 + "\n")

    print("1. Loading Processor...")
    processor_path = args.lora_dir if os.path.exists(args.lora_dir) else args.base_model
    processor = AutoProcessor.from_pretrained(processor_path)

    print("2. Loading Base Model in bfloat16...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if os.path.exists(args.lora_dir):
        print(f"3. Attaching Trained LoRA Weights from {args.lora_dir}...")
        model = PeftModel.from_pretrained(model, args.lora_dir)
        print("[SUCCESS] LoRA adapter attached successfully!")
    else:
        print("[INFO] No LoRA adapter found at path. Running with base model.")

    model.eval()

    def run_single_query(img_path, prompt_text_str, max_tokens=256):
        if not os.path.exists(img_path):
            print(f"[ERROR] Image not found: {img_path}")
            return
        image = Image.open(img_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt_text_str}
                ]
            }
        ]
        chat_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_prompt], images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print("\n🛰️ Generating answer on RTX 5060 Ti...")
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.2,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True
                )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        resp = processor.decode(generated_ids, skip_special_tokens=True).strip()
        print("\n" + "="*60)
        print("🛰️ SatQuery Model Output:")
        print("-" * 60)
        print(resp)
        print("="*60 + "\n")
        return resp

    if args.interactive:
        import random
        val_file = "data/qwen_val_multimodal.json" if os.path.exists("data/qwen_val_multimodal.json") else "data/qwen_val.json"
        with open(val_file, "r", encoding="utf-8") as f:
            val_samples = json.load(f)
        print("\n" + "="*60)
        print("[SatQuery] Interactive Satellite QA Mode Active!")
        print("Commands:")
        print("  - Type 'random' to test a random validation image & prompt")
        print("  - Type 'exit' or 'quit' to stop")
        print("  - Or enter an image path and custom question below")
        print("="*60 + "\n")

        current_img = args.image if (args.image and os.path.exists(args.image)) else val_samples[0]["image"]

        while True:
            try:
                user_in = input(f"\n[Image: {current_img}]\nEnter prompt (or 'random' / 'image <path>' / 'exit'): ").strip()
                if not user_in:
                    continue
                if user_in.lower() in ["exit", "quit", "q"]:
                    print("Exiting interactive mode.")
                    break
                if user_in.lower() == "random":
                    sample = random.choice(val_samples)
                    current_img = sample["image"]
                    q = sample["messages"][0]["content"][1]["text"]
                    gt = sample["messages"][1]["content"][0]["text"]
                    print(f"\n[Random Sample Selected: {sample['id']}]")
                    print(f"Task Type: {sample.get('task_type', 'QA')}")
                    print(f"Image: {current_img}")
                    print(f"Question: {q}")
                    print(f"Ground Truth Reference: {gt}")
                    run_single_query(current_img, q, max_tokens=args.max_tokens)
                elif user_in.lower().startswith("image "):
                    new_path = user_in[6:].strip().strip('"').strip("'")
                    if os.path.exists(new_path):
                        current_img = new_path
                        print(f"Switched active image to: {current_img}")
                    else:
                        print(f"File not found: {new_path}")
                else:
                    run_single_query(current_img, user_in, max_tokens=args.max_tokens)
            except KeyboardInterrupt:
                print("\nSession interrupted.")
                break
    else:
        # Single query mode
        image_path = args.image
        if not image_path or not os.path.exists(image_path):
            val_file = "data/qwen_val_multimodal.json" if os.path.exists("data/qwen_val_multimodal.json") else "data/qwen_val.json"
            with open(val_file, "r", encoding="utf-8") as f:
                val_samples = json.load(f)
            sample = val_samples[0]
            image_path = sample["image"]
            default_question = sample["messages"][0]["content"][1]["text"]
            ground_truth = sample["messages"][1]["content"][0]["text"]
            print(f"Using sample from validation set: {sample['id']}")
            print(f"Ground Truth Reference: {ground_truth}")
        else:
            default_question = "Analyze this satellite image. Describe what land cover types, surface features, and patterns are visible."

        question = args.prompt if args.prompt else default_question
        print(f"\nImage Path: {image_path}")
        print(f"Question: {question}\n")
        run_single_query(image_path, question, max_tokens=args.max_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Qwen3-VL satellite inference")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--lora_dir", type=str, default="output/qwen3_vl_satquery_multimodal_lora")
    parser.add_argument("--image", type=str, default=None, help="Path to satellite image PNG")
    parser.add_argument("--prompt", type=str, default=None, help="Custom question prompt")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--interactive", action="store_true", help="Launch continuous interactive chat session")

    args = parser.parse_args()
    test_inference(args)

