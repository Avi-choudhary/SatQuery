import os
from huggingface_hub import hf_hub_download
import pandas as pd

os.makedirs("data", exist_ok=True)

print("Downloading the BigEarthNet.txt annotations (~467 MB)...")

parquet_file = hf_hub_download(
    repo_id="BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
    filename="BigEarthNet.txt.parquet",
    repo_type="dataset",
    local_dir="data"
)

print(f"Downloaded successfully to: {parquet_file}")

# Load and inspect
df = pd.read_parquet(parquet_file)
print(f"Total annotations loaded: {len(df):,}")
print("Available task types:")
print(df["type"].value_counts())
