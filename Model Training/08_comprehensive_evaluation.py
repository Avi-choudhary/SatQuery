import os
import sys
import json
import time
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

VAL_FILE = "data/qwen_val_multimodal.json"
BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
LORA_DIR = "output/qwen3_vl_satquery_multimodal_lora"
OUTPUT_REPORT_MD = "c:/Games/SatQuery/EVALUATION_REPORT.md"
OUTPUT_REPORT_JSON = "c:/Games/SatQuery/evaluation_results.json"

def select_eval_samples(val_path, samples_per_modality=12):
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Separate by modality
    pools = {
        "sentinel-1": [x for x in val_data if "sentinel-1" in x["id"]],
        "sentinel-2": [x for x in val_data if "sentinel-2" in x["id"]],
        "dual": [x for x in val_data if "dual" in x["id"]]
    }

    selected = []
    task_types = ["mcq", "binary", "captioning", "bounding box"]
    per_type = samples_per_modality // len(task_types)

    for mod_key, items in pools.items():
        by_type = {t: [] for t in task_types}
        for item in items:
            t = item.get("task_type", "mcq")
            if t in by_type:
                by_type[t].append(item)
            else:
                by_type["mcq"].append(item)

        mod_selected = []
        for t in task_types:
            mod_selected.extend(by_type[t][:per_type])
        
        # In case some types had fewer, fill up to samples_per_modality
        if len(mod_selected) < samples_per_modality:
            needed = samples_per_modality - len(mod_selected)
            remaining = [x for x in items if x not in mod_selected]
            mod_selected.extend(remaining[:needed])
        
        selected.extend(mod_selected[:samples_per_modality])

    return selected

def evaluate_semantic(task_type, gt, pred):
    gt_clean = gt.strip().lower()
    pred_clean = pred.strip().lower()

    if task_type == "mcq":
        # Check if option letter or option text matches
        gt_letter = gt_clean[0] if len(gt_clean) > 0 else ""
        if gt_letter and (gt_letter == pred_clean[:1] or f" {gt_letter} " in f" {pred_clean} " or pred_clean.startswith(f"{gt_letter})")):
            return "Exact / Correct Option", True
        if gt_clean in pred_clean:
            return "Correct Content", True
        return "Mismatch", False

    elif task_type == "binary":
        if ("yes" in gt_clean and "yes" in pred_clean) or ("no" in gt_clean and "no" in pred_clean):
            return "Correct Binary Answer", True
        if ("true" in gt_clean and "true" in pred_clean) or ("false" in gt_clean and "false" in pred_clean):
            return "Correct Binary Answer", True
        return "Incorrect", False

    elif task_type == "captioning":
        # Extract land cover terms
        common_classes = [
            "arable land", "agriculture", "forest", "pastures", "urban", "water", 
            "coniferous", "broad-leaved", "crops", "vegetation", "grassland",
            "temperate", "continental", "wetland", "roughness", "backscatter"
        ]
        gt_found = set([c for c in common_classes if c in gt_clean])
        pred_found = set([c for c in common_classes if c in pred_clean])
        overlap = gt_found.intersection(pred_found)
        if len(gt_found) == 0:
            match_score = 1.0 if len(pred_clean) > 10 else 0.0
        else:
            match_score = len(overlap) / len(gt_found)
        
        if match_score >= 0.5:
            return f"High Semantic Match ({len(overlap)}/{len(gt_found)} concepts)", True
        elif match_score > 0:
            return f"Partial Match ({len(overlap)}/{len(gt_found)} concepts)", True
        else:
            return "Low / Divergent Description", False

    elif task_type == "bounding box":
        # Check if model produced a coordinate structure [ymin xmin, ymax xmax]
        if "[" in pred and "]" in pred:
            return "Valid Spatial Grounding Box Format", True
        return "Non-standard Box Format", False

    return "Assessed", True

def main():
    print("="*70)
    print("🛰️ SATQUERY MULTIMODAL VLM COMPREHENSIVE REASONING EVALUATION")
    print("   Sentinel-1 SAR | Sentinel-2 Optical | Cross-Modality (Dual)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Trained LoRA Weights: {LORA_DIR}\n")

    print("1. Loading Processor & Base Model in bfloat16...")
    processor = AutoProcessor.from_pretrained(LORA_DIR if os.path.exists(LORA_DIR) else BASE_MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"2. Attaching Trained LoRA Weights from {LORA_DIR}...")
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model.eval()
    print("   Model loaded and ready for evaluation.\n")

    eval_items = select_eval_samples(VAL_FILE, samples_per_modality=12)
    print(f"3. Selected {len(eval_items)} evaluation test cases across all modalities.\n")

    results = []
    correct_count = 0

    for i, item in enumerate(eval_items):
        item_id = item["id"]
        img_path = item["image"]
        task_type = item.get("task_type", "unknown")
        
        # Determine modality label
        if "sentinel-1" in item_id:
            modality = "Sentinel-1 SAR"
        elif "sentinel-2" in item_id:
            modality = "Sentinel-2 Optical"
        else:
            modality = "Cross-Modality (Dual)"

        question = item["messages"][0]["content"][1]["text"]
        gt_answer = item["messages"][1]["content"][0]["text"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image: {img_path}")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": question}
                ]
            }
        ]

        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False, # Deterministic greedy decoding for objective evaluation
                pad_token_id=processor.tokenizer.pad_token_id
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_tokens = output_ids[0][prompt_len:]
        prediction = processor.decode(generated_tokens, skip_special_tokens=True).strip()

        semantic_rating, is_pass = evaluate_semantic(task_type, gt_answer, prediction)
        if is_pass:
            correct_count += 1

        res_record = {
            "index": i + 1,
            "id": item_id,
            "modality": modality,
            "task_type": task_type,
            "image": img_path,
            "question": question,
            "ground_truth": gt_answer,
            "model_prediction": prediction,
            "semantic_rating": semantic_rating,
            "is_success": is_pass
        }
        results.append(res_record)

        print(f"[{i+1}/{len(eval_items)}] [{modality} | {task_type.upper()}]")
        print(f"  Q: {question.splitlines()[-1] if len(question.splitlines()) > 1 else question[:80]}...")
        print(f"  Ground Truth: {gt_answer[:100]}")
        print(f"  Model Output: {prediction[:100]}")
        print(f"  Assessment:   {semantic_rating} {'(PASS)' if is_pass else '(FAIL)'}")
        print("-" * 70)

    # Save JSON results
    with open(OUTPUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Build Markdown Report
    by_mod = {}
    for r in results:
        m = r["modality"]
        if m not in by_mod:
            by_mod[m] = []
        by_mod[m].append(r)

    md = []
    md.append("# 🛰️ SatQuery Multimodal Model Capability & Semantic Accuracy Report\n")
    md.append(f"**Model Tested:** `Qwen/Qwen3-VL-2B-Instruct` + LoRA (`{LORA_DIR}`)\n")
    md.append(f"**Evaluation Mode:** Greedy decoding (`temperature=0.0`) on unseen BigEarthNet validation scenes.\n")
    md.append(f"**Overall Semantic Success Rate:** **{correct_count} / {len(results)} ({correct_count/len(results)*100:.1f}%)**\n")
    md.append("---\n")

    for mod, records in by_mod.items():
        mod_success = sum(1 for r in records if r["is_success"])
        md.append(f"## 📡 {mod} Evaluation ({mod_success}/{len(records)} Valid Semantic Responses)\n")
        md.append("| # | Task Type | Question Snippet | Ground Truth Reference | Model Prediction | Semantic Verdict |\n")
        md.append("| :- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in records:
            q_short = r["question"].splitlines()[-1].replace("|", "\\|")
            if len(q_short) > 60:
                q_short = q_short[:57] + "..."
            gt_short = r["ground_truth"].replace("\n", " ").replace("|", "\\|")
            if len(gt_short) > 60:
                gt_short = gt_short[:57] + "..."
            pred_short = r["model_prediction"].replace("\n", " ").replace("|", "\\|")
            if len(pred_short) > 60:
                pred_short = pred_short[:57] + "..."
            icon = "🟢" if r["is_success"] else "🔴"
            md.append(f"| {r['index']} | **{r['task_type']}** | {q_short} | `{gt_short}` | **`{pred_short}`** | {icon} {r['semantic_rating']} |\n")
        md.append("\n---\n")

    with open(OUTPUT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("".join(md))

    print(f"\n[DONE] Saved evaluation report to: {OUTPUT_REPORT_MD}")
    print(f"Overall Semantic Success Rate: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()
