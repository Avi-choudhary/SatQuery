import os
import zipfile
import pandas as pd
import numpy as np

print("="*60)
print("[SatQuery] Step 1: Filter Full-Scale Disjoint Multimodal Annotations")
print("Ensuring 100% Zero-Duplicate QA Pairs Across S1, S2, and Dual-Modality")
print("="*60)

parquet_path = "data/BigEarthNet.txt.parquet"
if not os.path.exists(parquet_path):
    raise FileNotFoundError(f"Missing {parquet_path}. Please download it first.")

local_zip = "data/BigEarthNet_14K.zip"
if not os.path.exists(local_zip):
    raise FileNotFoundError(f"Missing {local_zip}.")

print(f"\n1. Indexing images from local archive {local_zip}...")
with zipfile.ZipFile(local_zip, "r") as zf:
    s1_all = set(
        os.path.splitext(os.path.basename(n))[0]
        for n in zf.namelist()
        if "BigEarthNet-S1" in n and n.endswith(".tif")
    )
    s2_all = set(
        os.path.splitext(os.path.basename(n))[0]
        for n in zf.namelist()
        if "BigEarthNet-S2" in n and n.endswith(".tif")
    )

print(f"Archive contains: {len(s1_all):,} Sentinel-1 SAR scenes | {len(s2_all):,} Sentinel-2 Optical scenes")

print(f"\n2. Loading annotations from {parquet_path}...")
df = pd.read_parquet(parquet_path)

# Filter to available scenes
df_avail = df[df["s1_name"].isin(s1_all) & df["patch_id"].isin(s2_all)].copy()
print(f"Total available annotations for archive scenes: {len(df_avail):,}")

# Shuffle all rows randomly to eliminate spatial or temporal ordering bias
df_shuffled = df_avail.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 3. Partition into 3 completely disjoint pools to guarantee zero question overlap
print("\n3. Creating mutually exclusive data pools for each modality...")
dual_pool = df_shuffled.iloc[:35000].copy()
s1_pool = df_shuffled.iloc[35000:150000].copy()
s2_pool = df_shuffled.iloc[150000:].copy()

print(f"Pool sizes: Dual = {len(dual_pool):,} | S1 = {len(s1_pool):,} | S2 = {len(s2_pool):,}")

# Helper to sample diverse tasks from a given pool
def sample_diverse_tasks(pool_df, target_count, modality_name, prompt_prefix, img_path_fn):
    n_mcq = int(target_count * 0.38)
    n_bin = int(target_count * 0.32)
    n_box = int(target_count * 0.18)
    n_cap = target_count - n_mcq - n_bin - n_box

    mcq_df = pool_df[pool_df["type"].isin(["mcq", "multiple-choice-qa"])]
    bin_df = pool_df[pool_df["type"] == "binary"]
    box_df = pool_df[pool_df["type"].str.contains("box|referring|grounding", case=False, na=False)]
    cap_df = pool_df[pool_df["type"].isin(["captioning", "image-captioning"])]

    s_mcq = mcq_df.sample(n=min(n_mcq, len(mcq_df)), random_state=42)
    s_bin = bin_df.sample(n=min(n_bin, len(bin_df)), random_state=42)
    s_box = box_df.sample(n=min(n_box, len(box_df)), random_state=42)
    s_cap = cap_df.sample(n=min(n_cap, len(cap_df)), random_state=42)

    selected = pd.concat([s_mcq, s_bin, s_box, s_cap])
    if len(selected) < target_count:
        used_ids = set(selected["ID"])
        leftover = pool_df[~pool_df["ID"].isin(used_ids)]
        needed = target_count - len(selected)
        extra = leftover.sample(n=min(needed, len(leftover)), random_state=42)
        selected = pd.concat([selected, extra])

    selected = selected.sample(frac=1.0, random_state=42).reset_index(drop=True)
    selected["modality"] = modality_name
    selected["prompt_prefix"] = prompt_prefix
    selected["image_path"] = selected.apply(img_path_fn, axis=1)
    return selected

print("\n4. Sampling Sentinel-1 SAR subset (55,000 QA pairs)...")
s1_sampled = sample_diverse_tasks(
    pool_df=s1_pool,
    target_count=55000,
    modality_name="sentinel-1",
    prompt_prefix="Analyze this Sentinel-1 Synthetic Aperture Radar (SAR) image (VV/VH backscatter composite).",
    img_path_fn=lambda r: f"data/images_s1/{r['s1_name']}.png"
)

print("5. Sampling Sentinel-2 Optical subset (48,000 QA pairs)...")
s2_sampled = sample_diverse_tasks(
    pool_df=s2_pool,
    target_count=48000,
    modality_name="sentinel-2",
    prompt_prefix="Analyze this Sentinel-2 optical satellite image.",
    img_path_fn=lambda r: f"data/images/{r['patch_id']}.png"
)

print("6. Sampling Dual-Modality subset (17,000 QA pairs)...")
dual_sampled = sample_diverse_tasks(
    pool_df=dual_pool,
    target_count=17000,
    modality_name="dual-modal",
    prompt_prefix="Analyze this dual-sensor satellite scene (Left: Sentinel-2 Optical RGB, Right: Sentinel-1 SAR radar backscatter).",
    img_path_fn=lambda r: f"data/images_dual/{r['patch_id']}__{r['s1_name']}.png"
)

# Merge all three completely disjoint subsets
full_df = pd.concat([s1_sampled, s2_sampled, dual_sampled]).sample(frac=1.0, random_state=42).reset_index(drop=True)

# Verification checks
total_unique_ids = full_df["ID"].nunique()
print("\n" + "="*60)
print("Integrity & Deduplication Verification:")
print(f"Total rows in dataset:       {len(full_df):,}")
print(f"Total unique annotation IDs: {total_unique_ids:,}")
assert len(full_df) == total_unique_ids, f"Error: Found {len(full_df) - total_unique_ids} duplicate IDs!"
print("Overlap between S1 and S2:   0 (Verified)")
print("Overlap between S1 and Dual: 0 (Verified)")
print("Overlap between S2 and Dual: 0 (Verified)")

output_parquet = "data/train_subset_multimodal.parquet"
full_df.to_parquet(output_parquet)
print(f"\nDeduplicated dataset successfully saved to: {output_parquet}")
print(f"\nModality Breakdown:")
print(full_df["modality"].value_counts(normalize=True).map(lambda v: f"{v*100:.1f}%") + " (" + full_df["modality"].value_counts().astype(str) + " samples)")
print(f"\nTask Type Breakdown:")
print(full_df.groupby(["modality", "type"]).size().unstack(fill_value=0))
print("="*60)
