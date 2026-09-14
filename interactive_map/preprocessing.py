"""
Model-Ready Radiometric Normalization.
Ensures zero distribution shift by applying the exact mathematical transformations
used during SatQuery training.
"""

from pathlib import Path
from typing import Optional
import numpy as np
from scipy.ndimage import uniform_filter
import rasterio


def linear_to_db(array: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Converts linear SAR backscatter amplitude to decibels (dB). Handles both raw DN and pre-calibrated values."""
    arr = array.astype(np.float32)
    # If uncalibrated raw digital numbers (DN > 10), convert to intensity using standard GRD factor
    if arr.max() > 10.0:
        arr = (arr / 500.0) ** 2
    return 10.0 * np.log10(np.clip(arr, epsilon, None))


def lee_filter(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Adaptive Lee filter for SAR speckle noise suppression.
    Smooths homogeneous areas while preserving high-contrast point targets and edges.
    """
    img_mean = uniform_filter(img, size=window_size)
    img_sqr_mean = uniform_filter(img**2, size=window_size)
    img_variance = np.maximum(0.0, img_sqr_mean - img_mean**2)
    overall_variance = float(np.var(img))
    weights = img_variance / (img_variance + overall_variance + 1e-10)
    return img_mean + weights * (img - img_mean)


def normalize_sar_band(img_db: np.ndarray, db_min: float = -25.0, db_max: float = 5.0) -> np.ndarray:
    """Normalizes SAR dB values to [0.0, 1.0] using fixed physical backscatter limits."""
    clipped = np.clip(img_db, db_min, db_max)
    return ((clipped - db_min) / (db_max - db_min)).astype(np.float32)


def normalize_optical(array: np.ndarray, scale_factor: float = 10000.0) -> np.ndarray:
    """Normalizes Sentinel-2 BOA surface reflectance values (0-10000) to [0.0, 1.0]."""
    normalized = array.astype(np.float32) / scale_factor
    return np.clip(normalized, 0.0, 1.0)


def preprocess_for_model(
    raw_geotiff_path: str,
    sensor_type: str = "sentinel-2",
    output_path: Optional[str] = None
) -> str:
    """
    Reads a raw fetched GeoTIFF, applies the appropriate sensor-specific normalization,
    and saves a model-ready float32 GeoTIFF.
    
    Args:
        raw_geotiff_path: Path to the raw downloaded GeoTIFF.
        sensor_type: "sentinel-2" or "sentinel-1".
        output_path: Optional destination path. Defaults to proc_{stem}.tif.
        
    Returns:
        Path to the saved normalized GeoTIFF.
    """
    input_file = Path(raw_geotiff_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {raw_geotiff_path}")

    if not output_path:
        output_file = input_file.parent / f"proc_{input_file.stem}.tif"
    else:
        output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(str(input_file)) as src:
        profile = src.profile.copy()
        raw_data = src.read()  # Shape: (bands, height, width)

        if sensor_type == "sentinel-2":
            proc_data = normalize_optical(raw_data)
        elif sensor_type == "sentinel-1":
            processed_bands = []
            for b in range(raw_data.shape[0]):
                band_db = linear_to_db(raw_data[b].astype(np.float32))
                band_filtered = lee_filter(band_db, window_size=5)
                band_norm = normalize_sar_band(band_filtered)
                processed_bands.append(band_norm)
            proc_data = np.stack(processed_bands, axis=0)
        else:
            proc_data = (raw_data.astype(np.float32) - raw_data.min()) / max(1e-6, float(raw_data.max() - raw_data.min()))

        profile.update(
            dtype=rasterio.float32,
            count=proc_data.shape[0],
            nodata=None
        )

        with rasterio.open(str(output_file), "w", **profile) as dst:
            dst.write(proc_data.astype(np.float32))

    return str(output_file.resolve())
