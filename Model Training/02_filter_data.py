import os
import zipfile
import fsspec
import pandas as pd

print("1. Inspecting available Sentinel-2 patches from BigEarthNet BEN-14K subset...")
BEN_ZIP_URL = "https://huggingface.co/datasets/ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2/resolve/main/BigEarthNet_14K.zip"

fs = fsspec.filesystem("http")
with fs.open(BEN_ZIP_URL) as f:
    zf = zipfile.ZipFile(f)
    s2_patch_map = {
        os.path.splitext(os.path.basename(n))[0]: n 
        for n in zf.namelist() 
        if "BigEarthNet-S2" in n and n.endswith(".tif")
    }

print(f"Found {len(s2_patch_map):,} Sentinel-2 patches available in BEN-14K.")

parquet_path = "data/BigEarthNet.txt.parquet"
if not os.path.exists(parquet_path):
    raise FileNotFoundError("Missing data/BigEarthNet.txt.parquet. Please run Step 1 first.")

print(f"2. Loading annotations from {parquet_path}...")
df = pd.read_parquet(parquet_path)

# Filter to available patches
df_avail = df[df["patch_id"].isin(s2_patch_map.keys())].copy()
print(f"Total annotations matching available S2 patches: {len(df_avail):,}")

# Option A: 5,000 patches with multi-task distribution
TARGET_PATCHES = 5000

patches_caps = set(df_avail[df_avail["type"].isin(["captioning", "image-captioning"])]["patch_id"].unique())
patches_box = set(df_avail[df_avail["type"].str.contains("box|referring|grounding", case=False, na=False)]["patch_id"].unique())
rich_patches = list(patches_caps.intersection(patches_box))

selected_patches = rich_patches[:TARGET_PATCHES]
if len(selected_patches) < TARGET_PATCHES:
    extra = list(patches_caps - set(selected_patches))[:TARGET_PATCHES - len(selected_patches)]
    selected_patches.extend(extra)

df_selected = df_avail[df_avail["patch_id"].isin(selected_patches)]

# Sample ~15,000 instruction pairs: 5,000 captions + 7,000 MCQ + 3,000 Grounding
vqa = df_selected[df_selected["type"].isin(["mcq", "multiple-choice-qa"])].sample(n=7000, random_state=42)
captions = df_selected[df_selected["type"].isin(["captioning", "image-captioning"])].sample(n=5000, random_state=42)
grounding = df_selected[df_selected["type"].str.contains("referring|box|grounding", case=False, na=False)].sample(n=3000, random_state=42)

subset_df = pd.concat([vqa, captions, grounding]).sample(frac=1.0, random_state=42).reset_index(drop=True)
subset_df["image_path"] = subset_df["patch_id"].apply(lambda pid: f"data/images/{pid}.png")

output_path = "data/train_subset_optionA.parquet"
subset_df.to_parquet(output_path)

unique_patches = subset_df["patch_id"].nunique()
print("\n" + "="*50)
print(f"Filtered dataset saved to: {output_path}")
print(f"Total instruction pairs: {len(subset_df)}")
print(f"Unique Sentinel-2 image patches required: {unique_patches}")
print("Task type distribution:")
print(subset_df["type"].value_counts())
print("="*50)
