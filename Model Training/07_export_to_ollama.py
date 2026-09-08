import os
import sys
import argparse
import subprocess
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def export_to_ollama(args):
    print("\n" + "="*60)
    print("[SatQuery] Export LoRA Model to Ollama Pipeline")
    print("="*60)
    print(f"Base Model: {args.base_model}")
    print(f"LoRA Adapter: {args.lora_dir}")
    print(f"Merged Output Dir: {args.merged_output_dir}")
    print(f"Ollama Target Tag: {args.ollama_tag}")
    print("="*60 + "\n")

    if not os.path.exists(args.lora_dir):
        raise FileNotFoundError(f"LoRA directory {args.lora_dir} not found. Please run training (Step 5) first.")

    os.makedirs(args.merged_output_dir, exist_ok=True)

    print("1. Loading Processor and Base Model in bfloat16...")
    processor = AutoProcessor.from_pretrained(args.base_model)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # Merge on CPU or GPU
    )

    print(f"2. Attaching and Merging LoRA Weights from {args.lora_dir}...")
    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    merged_model = model.merge_and_unload()

    print(f"3. Saving full merged model to: {args.merged_output_dir}...")
    merged_model.save_pretrained(args.merged_output_dir, max_shard_size="4GB")
    processor.save_pretrained(args.merged_output_dir)

    # Copy preprocessor_config.json from HF cache if not saved by processor
    import shutil, glob
    cache_preprocessors = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*/*preprocessor_config.json"))
    for src in cache_preprocessors:
        dst = os.path.join(args.merged_output_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

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

SYSTEM \"\"\"You are SatQuery, an advanced Earth Observation Vision-Language assistant specializing in Sentinel-2 satellite imagery analysis, terrain classification, and visual question answering for ISRO / Earth Observation applications.\"\"\"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.9
"""

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"4. Created Ollama Modelfile at: {modelfile_path}")

    # Register into Ollama
    print(f"\n5. Registering model into Ollama as '{args.ollama_tag}'...")
    try:
        cmd = ["ollama", "create", args.ollama_tag, "-f", modelfile_path, "--quantize", "q4_K_M"]
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[NOTE] Quantized creation returned code {result.returncode}, attempting standard creation...")
            cmd = ["ollama", "create", args.ollama_tag, "-f", modelfile_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        print("\n" + "="*60)
        print(f"[SUCCESS] Model '{args.ollama_tag}' is now registered in Ollama!")
        print(f"Run it directly from your terminal using:")
        print(f"  ollama run {args.ollama_tag}")
        print("="*60)
    except Exception as e:
        print(f"[NOTE] Automated ollama registration reported: {e}")
        print("You can manually register it anytime by running:")
        print(f"  ollama create {args.ollama_tag} -f \"{modelfile_path}\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA and export to Ollama")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--lora_dir", type=str, default="output/qwen3_vl_satquery_lora")
    parser.add_argument("--merged_output_dir", type=str, default="output/qwen3_vl_satquery_merged")
    parser.add_argument("--ollama_tag", type=str, default="satquery-qwen:2b")

    args = parser.parse_args()
    export_to_ollama(args)
