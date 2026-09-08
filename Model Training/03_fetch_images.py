import os
import io
import zipfile
import fsspec
import numpy as np
import pandas as pd
from PIL import Image
import tifffile
from tqdm import tqdm

BEN_ZIP_URL = "https://huggingface.co/datasets/ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2/resolve/main/BigEarthNet_14K.zip"

parquet_file = "data/train_subset_optionA.parquet"
if not os.path.exists(parquet_file):
    parquet_file = "data/train_subset_3k.parquet"

if not os.path.exists(parquet_file):
    raise FileNotFoundError("No parquet dataset found. Please run Step 2 first.")

print(f"Loading target patches from: {parquet_file}")
df = pd.read_parquet(parquet_file)
target_patches = list(df["patch_id"].unique())

images_dir = "data/images"
os.makedirs(images_dir, exist_ok=True)
print(f"Total unique Sentinel-2 image patches required: {len(target_patches)}")

print("Connecting to remote BigEarthNet BEN-14K archive via HTTP range streaming...")
fs = fsspec.filesystem("http")

with fs.open(BEN_ZIP_URL) as f:
    zf = zipfile.ZipFile(f)
    name_to_zip_path = {
        os.path.splitext(os.path.basename(n))[0]: n
        for n in zf.namelist()
        if "BigEarthNet-S2" in n and n.endswith(".tif")
    }

    saved_count = 0
    skipped_count = 0

    for idx, patch_id in enumerate(tqdm(target_patches, desc="Streaming BigEarthNet patches")):
        save_path = os.path.join(images_dir, f"{patch_id}.png")
        seq_path = os.path.join(images_dir, f"patch_{idx:04d}.png")

        if os.path.exists(save_path):
            skipped_count += 1
            continue

        if patch_id not in name_to_zip_path:
            continue

        zip_entry = name_to_zip_path[patch_id]
        raw_data = zf.read(zip_entry)
        
        # Read multispectral Sentinel-2 TIFF (120x120)
        arr = tifffile.imread(io.BytesIO(raw_data))

        # Sentinel-2 True Color RGB:
        # Band 2: B04 (Red), Band 1: B03 (Green), Band 0: B02 (Blue)
        rgb = arr[[2, 1, 0], :, :].transpose(1, 2, 0).astype("float32")

        # 2% to 98% percentile contrast stretch
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype("uint8")

        img = Image.fromarray(rgb_norm)
        img.save(save_path)

        if not os.path.exists(seq_path):
            img.save(seq_path)

        saved_count += 1

total_files = len([f for f in os.listdir(images_dir) if f.endswith(".png")])
total_bytes = sum(os.path.getsize(os.path.join(images_dir, f)) for f in os.listdir(images_dir) if f.endswith(".png"))
total_mb = total_bytes / (1024 * 1024)

print("\n" + "="*50)
print(f"Extraction complete! Newly saved: {saved_count} | Already cached: {skipped_count}")
print(f"Total files in {images_dir}/: {total_files}")
print(f"Total disk used: {total_mb:.1f} MB")
print("="*50)
