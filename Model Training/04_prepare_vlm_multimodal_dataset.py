import os
import json
import pandas as pd

print("="*60)
print("[SatQuery] Step 3: Prepare Full-Scale Multimodal VLM Dataset for Qwen3-VL-2B")
print("Target: ~120,000 Total QA Pairs (108,000 Train / 12,000 Validation)")
print("="*60)

parquet_path = "data/train_subset_multimodal.parquet"
if not os.path.exists(parquet_path):
    raise FileNotFoundError("Missing data/train_subset_multimodal.parquet. Please run Step 1 first.")

print(f"\n1. Loading annotations from {parquet_path}...")
df = pd.read_parquet(parquet_path)
print(f"Total entries loaded: {len(df):,}")

# 2. Fast set-based image verification
print("\n2. Verifying image files on disk across modalities...")
unique_images = df["image_path"].unique()
existing_images = set(p for p in unique_images if os.path.exists(p))
missing_images = set(unique_images) - existing_images

if missing_images:
    print(f"Warning: {len(missing_images):,} unique image files missing from disk. Filtering rows...")
    df = df[df["image_path"].isin(existing_images)].reset_index(drop=True)
    print(f"Retained {len(df):,} rows with verified images.")
else:
    print(f"All {len(unique_images):,} unique image assets successfully verified on disk!")

# Shuffle dataset
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 90% train, 10% validation split
val_size = int(len(df) * 0.10)
train_df = df.iloc[:-val_size].reset_index(drop=True)
val_df = df.iloc[-val_size:].reset_index(drop=True)

def convert_to_qwen_multimodal_format(dataframe, split_name):
    records = []
    for idx, row in dataframe.iterrows():
        img_rel_path = row["image_path"].replace("\\", "/")
        prefix = row.get("prompt_prefix", "Analyze this satellite image.")
        user_prompt = f"{prefix}\n\nQuestion: {row['input']}"
        
        # Globally unique ID using parquet annotation ID and split
        sample_id = f"ben_{split_name}_{row['ID']}_{row['modality']}"
        
        record = {
            "id": sample_id,
            "image": img_rel_path,
            "modality": row["modality"],
            "task_type": row.get("type", "unknown"),
            "category": row.get("category", "unknown"),
            "patch_id": row.get("patch_id", ""),
            "s1_name": row.get("s1_name", ""),
            # ShareGPT / LLaMA-Factory VLM format
            "conversations": [
                {
                    "from": "user",
                    "value": f"<image>\n{user_prompt}"
                },
                {
                    "from": "assistant",
                    "value": str(row["output"])
                }
            ],
            # Standard HuggingFace Qwen3-VL chat messages format
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_rel_path},
                        {"type": "text", "text": user_prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": str(row["output"])}
                    ]
                }
            ]
        }
        records.append(record)
    return records

print("\n3. Converting to Qwen3-VL chat structure...")
train_data = convert_to_qwen_multimodal_format(train_df, "train")
val_data = convert_to_qwen_multimodal_format(val_df, "val")

train_json_path = "data/qwen_train_multimodal.json"
val_json_path = "data/qwen_val_multimodal.json"

print(f"Writing {len(train_data):,} train samples to {train_json_path}...")
with open(train_json_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

print(f"Writing {len(val_data):,} val samples to {val_json_path}...")
with open(val_json_path, "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print(f"Full-Scale Dataset Preparation Complete!")
print(f"Train samples saved to {train_json_path}: {len(train_data):,}")
print(f"Validation samples saved to {val_json_path}: {len(val_data):,}")
print("\nTrain Set Modality Breakdown:")
print(train_df["modality"].value_counts(normalize=True).map(lambda v: f"{v*100:.1f}%") + " (" + train_df["modality"].value_counts().astype(str) + ")")
print("\nTrain Set Task Type Breakdown:")
print(train_df.groupby(["modality", "type"]).size().unstack(fill_value=0))
print("="*60)
