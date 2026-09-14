import os
import sys
import argparse
import subprocess
import shutil
import glob
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def export_to_ollama(args):
    # Set default paths if not provided
    if not args.lora_dir:
        if args.epoch == 2:
            if os.path.exists("output/epoch2_golden_checkpoint"):
                args.lora_dir = "output/epoch2_golden_checkpoint"
            else:
                args.lora_dir = "output/qwen3_vl_satquery_multimodal_lora"
        elif args.epoch == 3:
            args.lora_dir = "output/qwen3_vl_satquery_multimodal_lora/checkpoint-epoch-3"
        else:
            args.lora_dir = "output/qwen3_vl_satquery_multimodal_lora"

    if not args.merged_output_dir:
        args.merged_output_dir = "output/qwen3_vl_satquery_merged"

    if not args.ollama_tag:
        args.ollama_tag = "satquery-qwen:2b"

    print("\n" + "="*65)
    print("[SatQuery] Export Multimodal Model to Ollama & Backend")
    print("="*65)
    print(f"Base Model:         {args.base_model}")
    print(f"LoRA Adapter:       {args.lora_dir}")
    print(f"Merged Output Dir:  {args.merged_output_dir}")
    print(f"Primary Ollama Tag: {args.ollama_tag}")
    print("="*65 + "\n")

    if not os.path.exists(args.lora_dir):
        raise FileNotFoundError(
            f"LoRA directory '{args.lora_dir}' not found. Please ensure the checkpoint exists."
        )

    os.makedirs(args.merged_output_dir, exist_ok=True)

    print("1. Loading Processor and Base Model in bfloat16 on CPU...")
    processor = AutoProcessor.from_pretrained(args.base_model)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )

    print(f"2. Attaching and Merging LoRA Weights from {args.lora_dir}...")
    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    merged_model = model.merge_and_unload()

    print(f"3. Saving full merged model to: {args.merged_output_dir}...")
    merged_model.save_pretrained(args.merged_output_dir, max_shard_size="4GB")
    processor.save_pretrained(args.merged_output_dir)

    # Copy preprocessor configs from HF cache if not saved by processor
    cache_preprocessors = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*/*preprocessor_config.json")
    )
    for src in cache_preprocessors:
        dst = os.path.join(args.merged_output_dir, os.path.basename(src))
        if not os.path.exists(dst):
            try:
                shutil.copyfile(src, dst)
            except Exception:
                pass

    # Generate Ollama Modelfile
    modelfile_path = os.path.join(args.merged_output_dir, "Modelfile")
    merged_abs_path = os.path.abspath(args.merged_output_dir).replace("\\", "/")

    modelfile_content = f"""FROM {merged_abs_path}

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>\"\"\"

SYSTEM \"\"\"You are SatQuery, an advanced Earth Observation Vision-Language assistant specializing in Sentinel-1 SAR radar imagery, Sentinel-2 optical imagery, cross-modality satellite scene understanding, terrain classification, and visual question answering for ISRO / Earth Observation applications.\"\"\"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.9
"""

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"4. Created Ollama Modelfile at: {modelfile_path}")

    # Register into Ollama under both primary tag and qwen3-vl:2b for complete integration
    tags_to_register = [args.ollama_tag]
    if "qwen3-vl:2b" not in tags_to_register:
        tags_to_register.append("qwen3-vl:2b")

    # Prepare environment for Ollama CLI
    ollama_env = os.environ.copy()
    default_ollama_path = os.path.expanduser("~/.ollama/models")
    if os.path.exists(default_ollama_path):
        ollama_env["OLLAMA_MODELS"] = default_ollama_path

    for tag in tags_to_register:
        print(f"\n5. Registering model into Ollama as '{tag}'...")
        try:
            # 1. First attempt: Experimental safetensors import (preserves full bfloat16 fidelity)
            cmd = ["ollama", "create", tag, "-f", modelfile_path, "--experimental"]
            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=ollama_env, capture_output=True, text=True)
            if result.returncode != 0:
                # 2. Fallback attempt: Standard creation
                print(f"[NOTE] Experimental creation returned code {result.returncode}, attempting standard creation...")
                cmd = ["ollama", "create", tag, "-f", modelfile_path]
                result = subprocess.run(cmd, env=ollama_env, capture_output=True, text=True, check=True)
            print(result.stdout)
            print(f"[SUCCESS] Model '{tag}' is now registered in Ollama!")
        except Exception as e:
            print(f"[NOTE] Automated ollama registration for {tag} reported: {e}")
            print("You can manually register it anytime by running:")
            print(f"  ollama create {tag} -f \"{modelfile_path}\" --experimental")

    print("\n" + "="*65)
    print("[EXPORT & INTEGRATION COMPLETE]")
    print(f"1. Merged Model: {args.merged_output_dir}")
    print("2. Ollama Tags:  satquery-qwen:2b  and  qwen3-vl:2b")
    print("3. Test with:    ollama run satquery-qwen:2b")
    print("="*65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA and export to Ollama with full integration")
    parser.add_argument("--epoch", type=int, choices=[2, 3], default=2, help="Select Epoch 2 (Golden) or Epoch 3 model to export")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--lora_dir", type=str, default=None, help="Explicit path to LoRA directory")
    parser.add_argument("--merged_output_dir", type=str, default=None, help="Explicit path to merged output directory")
    parser.add_argument("--ollama_tag", type=str, default="satquery-qwen:2b", help="Target Ollama tag")

    args = parser.parse_args()
    export_to_ollama(args)
