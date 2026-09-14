import os
import io
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import tifffile
from tqdm import tqdm

print("="*60)
print("[SatQuery] Step 2: Extract Full Multimodal Satellite Imagery")
print("Extracting All Required Sentinel-1 SAR, Sentinel-2 Optical & Dual Composites")
print("="*60)

parquet_file = "data/train_subset_multimodal.parquet"
if not os.path.exists(parquet_file):
    raise FileNotFoundError("Missing data/train_subset_multimodal.parquet. Please run Step 1 first.")

print(f"\n1. Loading dataset targets from: {parquet_file}")
df = pd.read_parquet(parquet_file)

# Required targets
target_s1 = set(df[df["modality"].isin(["sentinel-1", "dual-modal"])]["s1_name"].unique())
target_s2 = set(df[df["modality"].isin(["sentinel-2", "dual-modal"])]["patch_id"].unique())
target_dual = df[df["modality"] == "dual-modal"][["patch_id", "s1_name"]].drop_duplicates().values.tolist()

s1_dir = "data/images_s1"
s2_dir = "data/images"
dual_dir = "data/images_dual"

os.makedirs(s1_dir, exist_ok=True)
os.makedirs(s2_dir, exist_ok=True)
os.makedirs(dual_dir, exist_ok=True)

# Check what is already extracted
existing_s1 = set(f.replace(".png", "") for f in os.listdir(s1_dir) if f.endswith(".png"))
existing_s2 = set(f.replace(".png", "") for f in os.listdir(s2_dir) if f.endswith(".png"))
existing_dual = set(f.replace(".png", "") for f in os.listdir(dual_dir) if f.endswith(".png"))

to_extract_s1 = [s for s in target_s1 if s not in existing_s1]
to_extract_s2 = [s for s in target_s2 if s not in existing_s2]
to_generate_dual = [(p, s) for p, s in target_dual if f"{p}__{s}" not in existing_dual]

print(f"Sentinel-1 SAR:     {len(existing_s1):,} on disk | {len(to_extract_s1):,} to extract")
print(f"Sentinel-2 Optical: {len(existing_s2):,} on disk | {len(to_extract_s2):,} to extract")
print(f"Dual Composites:    {len(existing_dual):,} on disk | {len(to_generate_dual):,} to generate")

# SAR Preprocessing (VV, VH, VV-VH composite with percentile contrast stretch)
def sar_to_false_color_rgb(arr_2band):
    vv = arr_2band[0]
    vh = arr_2band[1]
    diff = vv - vh

    def norm_ch(ch):
        p2, p98 = np.percentile(ch, (2, 98))
        if p98 - p2 < 1e-4:
            return np.zeros_like(ch, dtype=np.uint8)
        stretched = np.clip((ch - p2) / (p98 - p2) * 255.0, 0, 255)
        return stretched.astype(np.uint8)

    r = norm_ch(vv)
    g = norm_ch(vh)
    b = norm_ch(diff)
    return np.stack([r, g, b], axis=-1)

# Optical Preprocessing (B04, B03, B02 RGB true color with percentile stretch)
def s2_to_rgb(arr_multi):
    rgb = arr_multi[[2, 1, 0], :, :].transpose(1, 2, 0).astype("float32")
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype("uint8")
    return rgb_norm

local_zip = "data/BigEarthNet_14K.zip"
if not os.path.exists(local_zip):
    raise FileNotFoundError(f"Missing {local_zip}.")

print(f"\n2. Opening local archive {local_zip}...")
with zipfile.ZipFile(local_zip, "r") as zf:
    s1_map = {
        os.path.splitext(os.path.basename(n))[0]: n
        for n in zf.namelist()
        if "BigEarthNet-S1" in n and n.endswith(".tif")
    }
    s2_map = {
        os.path.splitext(os.path.basename(n))[0]: n
        for n in zf.namelist()
        if "BigEarthNet-S2" in n and n.endswith(".tif")
    }

    # 1. Extract Sentinel-1 SAR
    if to_extract_s1:
        print(f"\nExtracting {len(to_extract_s1):,} Sentinel-1 SAR patches...")
        saved_s1 = 0
        for s1_name in tqdm(to_extract_s1, desc="Extracting S1 SAR"):
            out_path = os.path.join(s1_dir, f"{s1_name}.png")
            if s1_name not in s1_map:
                continue
            raw_bytes = zf.read(s1_map[s1_name])
            arr = tifffile.imread(io.BytesIO(raw_bytes))
            rgb_arr = sar_to_false_color_rgb(arr)
            Image.fromarray(rgb_arr).save(out_path)
            saved_s1 += 1
        print(f"Saved {saved_s1:,} new Sentinel-1 SAR patches.")
    else:
        print("\nAll Sentinel-1 SAR patches are already extracted on disk.")

    # 2. Extract Sentinel-2 Optical
    if to_extract_s2:
        print(f"\nExtracting {len(to_extract_s2):,} Sentinel-2 Optical patches...")
        saved_s2 = 0
        for patch_id in tqdm(to_extract_s2, desc="Extracting S2 Optical"):
            out_path = os.path.join(s2_dir, f"{patch_id}.png")
            if patch_id not in s2_map:
                continue
            raw_bytes = zf.read(s2_map[patch_id])
            arr = tifffile.imread(io.BytesIO(raw_bytes))
            rgb_arr = s2_to_rgb(arr)
            Image.fromarray(rgb_arr).save(out_path)
            saved_s2 += 1
        print(f"Saved {saved_s2:,} new Sentinel-2 Optical patches.")
    else:
        print("\nAll Sentinel-2 Optical patches are already extracted on disk.")

# 3. Generate Dual-Modality Side-by-Side Composites
if to_generate_dual:
    print(f"\n3. Generating {len(to_generate_dual):,} Dual-Modality Composites (240x120)...")
    saved_dual = 0
    for patch_id, s1_name in tqdm(to_generate_dual, desc="Generating Dual Composites"):
        out_dual = os.path.join(dual_dir, f"{patch_id}__{s1_name}.png")
        s2_file = os.path.join(s2_dir, f"{patch_id}.png")
        s1_file = os.path.join(s1_dir, f"{s1_name}.png")

        if not os.path.exists(s2_file) or not os.path.exists(s1_file):
            continue

        img_s2 = Image.open(s2_file).convert("RGB")
        img_s1 = Image.open(s1_file).convert("RGB")

        composite = Image.new("RGB", (img_s2.width + img_s1.width, max(img_s2.height, img_s1.height)))
        composite.paste(img_s2, (0, 0))
        composite.paste(img_s1, (img_s2.width, 0))
        composite.save(out_dual)
        saved_dual += 1
    print(f"Generated {saved_dual:,} new Dual-Modality composites.")
else:
    print("\n3. All required Dual-Modality composites are already generated on disk.")

print("\n" + "="*60)
print("Final Image Asset Inventory on Disk:")
print(f"Sentinel-1 SAR patches in {s1_dir}/:     {len(os.listdir(s1_dir)):,}")
print(f"Sentinel-2 Optical patches in {s2_dir}/: {len(os.listdir(s2_dir)):,}")
print(f"Dual-Modality composites in {dual_dir}/: {len(os.listdir(dual_dir)):,}")
print("="*60)
