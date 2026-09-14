"""
Configuration for the Interactive Map STAC + COG imagery fetcher.
"""

from pathlib import Path

# STAC API endpoint
STAC_API_URL = "https://earth-search.aws.element84.com/v1"

# Supported collection IDs
COLLECTIONS = {
    "sentinel-2": "sentinel-2-l2a",        # Primary: Sentinel-2 Collection 1 L2A (AWS Open Data)
    "sentinel-2-fallback": "sentinel-2-c1-l2a",
    "sentinel-1": "sentinel-1-grd"         # Sentinel-1 Level-1C Ground Range Detected
}

# Band mappings (matching model training & gis_extraction_logic)
# Sentinel-2 bands: B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)
S2_BANDS = {
    "red": "red",       # B04
    "green": "green",   # B03
    "blue": "blue",     # B02
    "nir": "nir"        # B08
}

# Sentinel-1 polarizations: VV, VH
S1_POLARIZATIONS = ["vv", "vh"]

# Constraints & Safety Limits
MAX_AOI_AREA_SQ_KM = 2500.0   # ~50 km x 50 km maximum box
DEFAULT_MAX_CLOUD_COVER = 20  # Percentage (0-100)
DEFAULT_SEARCH_DAYS = 90      # Look back 90 days by default

# Target directories
MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
FETCHED_DIR = DATA_DIR / "fetched"
FETCHED_DIR.mkdir(parents=True, exist_ok=True)
