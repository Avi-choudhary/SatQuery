from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

RAW_S1_DIR = DATA_DIR / "raw" / "sentinel1"      # raw SAR files go here
RAW_S2_DIR = DATA_DIR / "raw" / "sentinel2"      # raw optical files go here
INTERIM_DIR = DATA_DIR / "interim"
PATCHES_DIR = DATA_DIR / "processed" / "patches"
PREVIEWS_DIR = DATA_DIR / "processed" / "previews"
METADATA_DIR = DATA_DIR / "processed" / "metadata"

TARGET_CRS = "EPSG:32643"     # UTM zone 43N (metric) so resolution=10 means 10 meters
WGS84_CRS = "EPSG:4326"       # Standard Lat/Lon for web mapping and LLM metadata
TARGET_RESOLUTION = 10        # 10 metres per pixel — Sentinel-2's sharpest bands

S2_BANDS = ["B02", "B03", "B04", "B08"]   # Blue, Green, Red, Near-Infrared
S1_POLARIZATIONS = ["VV", "VH"]

PATCH_SIZE = 256          # each final tile is 256x256 pixels
PATCH_STRIDE = 256        # no overlap between tiles

MAX_PAIR_GAP_DAYS = 5     # optical & SAR must be within 5 days to count as "the same moment"