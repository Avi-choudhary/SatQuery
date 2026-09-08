import os
import json
import pandas as pd

parquet_path = "data/train_subset_optionA.parquet"
if not os.path.exists(parquet_path):
    parquet_path = "data/train_subset_3k.parquet"

if not os.path.exists(parquet_path):
    raise FileNotFoundError("No parquet dataset found. Please run Step 2 first.")

print(f"Loading annotations from {parquet_path}...")
df = pd.read_parquet(parquet_path)

# Verify all images exist
print("Verifying image file existence...")
missing = []
for p in df["image_path"].unique():
    if not os.path.exists(p):
        missing.append(p)

if missing:
    print(f"Warning: {len(missing)} images missing from data/images/. Filtering out missing entries.")
    df = df[~df["image_path"].isin(missing)].reset_index(drop=True)
else:
    print(f"All {df['image_path'].nunique()} unique images successfully verified on disk!")

# Shuffle dataset
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 90% train, 10% validation split
val_size = int(len(df) * 0.10)
train_df = df.iloc[:-val_size].reset_index(drop=True)
val_df = df.iloc[-val_size:].reset_index(drop=True)

def convert_to_qwen_format(dataframe):
    records = []
    for idx, row in dataframe.iterrows():
        img_rel_path = row["image_path"].replace("\\", "/")
        # System prompt setting context as an Earth Observation model
        user_prompt = f"Analyze this Sentinel-2 satellite image.\n\nQuestion: {row['input']}"
        
        record = {
            "id": f"ben_{row['patch_id']}_{idx:05d}",
            "image": img_rel_path,
            "task_type": row.get("type", "unknown"),
            "category": row.get("category", "unknown"),
            # Standard ShareGPT / LLaMA-Factory VLM format
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
            # Standard HuggingFace Qwen-VL chat messages format
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

train_data = convert_to_qwen_format(train_df)
val_data = convert_to_qwen_format(val_df)

train_json_path = "data/qwen_train.json"
val_json_path = "data/qwen_val.json"

with open(train_json_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open(val_json_path, "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)

print("\n" + "="*50)
print(f"Dataset preparation complete!")
print(f"Train samples saved to {train_json_path}: {len(train_data):,}")
print(f"Validation samples saved to {val_json_path}: {len(val_data):,}")
print("Task type breakdown in train set:")
print(train_df["type"].value_counts())
print("="*50)
