import os
import sys
from pathlib import Path

# Add project root to sys.path so 'src' and 'config' can be imported cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Automatically locate proj.db within your virtual environment
for root, dirs, files in os.walk(sys.prefix):
    if "proj.db" in files:
        os.environ["PROJ_LIB"] = root
        break

# Now import geospatial and project modules safely
from src.coregistration import coregister_pair
from src.io_utils import load_raster, save_raster
from src.optical_processing import process_optical_tile
from src.sar_processing import lee_filter, linear_to_db
from src.tiling import tile_raster


def run():
  # 1. Process Optical Tile (Sentinel-2)
  input_path_s2 = "data/raw/sentinel2/delhi_20260112_S2.tif"
  output_path_s2 = "data/interim/delhi_20260112_S2_processed.tif"
  if os.path.exists(input_path_s2):
    print(f"Loading raw optical image from {input_path_s2}...")
    array_s2, profile_s2 = load_raster(input_path_s2)
    print("Processing optical tile (normalizing reflectance)...")
    processed_s2 = process_optical_tile(array_s2)
    save_raster(output_path_s2, processed_s2, profile_s2)
    print(f"Successfully saved processed optical tile to {output_path_s2}!")
  elif os.path.exists(output_path_s2):
    print(f"Loading existing processed optical image from {output_path_s2}...")
    processed_s2, profile_s2 = load_raster(output_path_s2)
  else:
    raise FileNotFoundError(f"Neither {input_path_s2} nor {output_path_s2} found.")

  # 2. Process SAR Tile (Sentinel-1)
  input_path_s1 = "data/raw/sentinel1/delhi_20260112_S1.tif"
  output_path_s1 = "data/interim/delhi_20260112_S1_processed.tif"
  if os.path.exists(input_path_s1):
    print(f"\nLoading raw SAR image from {input_path_s1}...")
    array_s1, profile_s1 = load_raster(input_path_s1)
    print("Converting SAR linear values to decibels (dB)...")
    db_s1 = linear_to_db(array_s1)
    print("Applying Lee filter to remove speckle noise...")
    filtered_s1 = lee_filter(db_s1)
    save_raster(output_path_s1, filtered_s1, profile_s1)
    print(f"Successfully saved processed SAR tile to {output_path_s1}!")
  elif os.path.exists(output_path_s1):
    print(f"\nLoading existing processed SAR image from {output_path_s1}...")
    filtered_s1, profile_s1 = load_raster(output_path_s1)
  else:
    raise FileNotFoundError(f"Neither {input_path_s1} nor {output_path_s1} found.")

  # 3. Co-registration (Aligning Optical and SAR)
  print(
      "\nPerforming spatial co-registration to align optical and SAR grids..."
  )
  aligned_s2, aligned_s1, target_profile = coregister_pair(
      processed_s2, profile_s2, filtered_s1, profile_s1
  )

  aligned_path_s2 = "data/interim/delhi_20260112_S2_aligned.tif"
  aligned_path_s1 = "data/interim/delhi_20260112_S1_aligned.tif"
  save_raster(aligned_path_s2, aligned_s2, target_profile)
  save_raster(aligned_path_s1, aligned_s1, target_profile)
  print(
      "Successfully aligned and saved rasters to data/interim/ with identical"
      " grid profiles!"
  )

  # 4. Patch Generation (Tiling aligned images)
  print(
      "\nGenerating 256x256 image patches for downstream machine learning"
      " tasks..."
  )

  # Load the aligned optical and SAR images
  aligned_s2_array, aligned_profile = load_raster(
      "data/interim/delhi_20260112_S2_aligned.tif"
  )
  aligned_s1_array, _ = load_raster(
      "data/interim/delhi_20260112_S1_aligned.tif"
  )

  # Run tile_raster for both sensors
  tile_raster(
      aligned_s2_array, aligned_profile, "delhi", "20260112", "S2"
  )
  tile_raster(
      aligned_s1_array, aligned_profile, "delhi", "20260112", "S1"
  )

  print(
      "Pipeline complete! Patches and metadata cards saved to data/processed/."
  )


if __name__ == "__main__":
  run()