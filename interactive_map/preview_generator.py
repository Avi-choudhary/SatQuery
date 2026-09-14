"""
Preview Generator for Satellite Rasters.
Produces high-quality, contrast-stretched web preview PNGs from GeoTIFFs.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
import rasterio


def _stretch_channel(arr: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
    """Stretches a 2D numeric array to uint8 (0-255) using percentile clipping."""
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
        
    p_low, p_high = np.percentile(valid, (lower_pct, upper_pct))
    if p_high <= p_low:
        p_low, p_high = float(valid.min()), float(valid.max())
        
    denom = max(1e-6, float(p_high - p_low))
    clipped = np.clip(arr, p_low, p_high)
    scaled = (clipped - p_low) / denom * 255.0
    return np.nan_to_num(scaled, nan=0.0).astype(np.uint8)


def generate_preview(
    geotiff_path: str,
    sensor_type: str = "sentinel-2",
    output_path: Optional[str] = None
) -> str:
    """
    Generates a web-friendly PNG thumbnail preview with contrast enhancement.
    
    Args:
        geotiff_path: Path to the GeoTIFF file.
        sensor_type: "sentinel-2" or "sentinel-1".
        output_path: Optional path for the generated PNG. Defaults to prev_{stem}.png in same folder.
        
    Returns:
        Absolute path to the created PNG file.
    """
    gtiff_file = Path(geotiff_path)
    if not gtiff_file.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {geotiff_path}")

    if not output_path:
        output_file = gtiff_file.parent / f"preview_{gtiff_file.stem}.png"
    else:
        output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(str(gtiff_file)) as src:
        band_count = src.count
        
        if sensor_type == "sentinel-2":
            # For Sentinel-2: Band 1=Red, Band 2=Green, Band 3=Blue
            if band_count >= 3:
                r = src.read(1)
                g = src.read(2)
                b = src.read(3)
                rgb = np.stack([
                    _stretch_channel(r, 2.0, 98.0),
                    _stretch_channel(g, 2.0, 98.0),
                    _stretch_channel(b, 2.0, 98.0)
                ], axis=-1)
            else:
                gray = src.read(1)
                stretched = _stretch_channel(gray, 2.0, 98.0)
                rgb = np.stack([stretched, stretched, stretched], axis=-1)
                
            img = Image.fromarray(rgb)
            img.save(str(output_file), "PNG")
            
        elif sensor_type == "sentinel-1":
            # For Sentinel-1 SAR: Convert linear amplitude to dB first
            # Band 1=VV, Band 2=VH (if 2 bands)
            vv = src.read(1).astype(np.float32)
            vv_db = 10.0 * np.log10(np.clip(vv, 1e-6, None))
            
            if band_count >= 2:
                vh = src.read(2).astype(np.float32)
                vh_db = 10.0 * np.log10(np.clip(vh, 1e-6, None))
                
                # False-color composite: Red=VV, Green=VH, Blue=VV-VH ratio (in dB)
                vv_norm = _stretch_channel(vv_db, 5.0, 95.0)
                vh_norm = _stretch_channel(vh_db, 5.0, 95.0)
                ratio_norm = _stretch_channel(vv_db - vh_db, 5.0, 95.0)
                rgb = np.stack([vv_norm, vh_norm, ratio_norm], axis=-1)
            else:
                stretched = _stretch_channel(vv_db, 5.0, 95.0)
                rgb = np.stack([stretched, stretched, stretched], axis=-1)
                
            img = Image.fromarray(rgb)
            img.save(str(output_file), "PNG")
            
        else:
            # Fallback generic grayscale
            gray = src.read(1)
            stretched = _stretch_channel(gray, 2.0, 98.0)
            img = Image.fromarray(stretched)
            img.save(str(output_file), "PNG")

    return str(output_file.resolve())


WEB_MERCATOR = "EPSG:3857"


def generate_web_overlay(
    geotiff_path: str,
    sensor_type: str = "sentinel-2",
    output_dir: Optional[str] = None,
    max_dim: int = 2048,
    suffix: str = "web"
) -> Optional[Dict[str, Any]]:
    """
    Renders a georeferenced raster reprojected to Web Mercator (EPSG:3857) as an RGBA PNG.
    Matches MapLibre GL's Web Mercator pixel grid perfectly with transparent nodata skirts,
    preventing any shearing or spatial misregistration.
    """
    gtiff_file = Path(geotiff_path)
    if not gtiff_file.exists():
        return None

    try:
        from rasterio.warp import (
            Resampling,
            calculate_default_transform,
            reproject,
            transform_bounds,
        )

        with rasterio.open(str(gtiff_file)) as src:
            if src.crs is None:
                return None

            transform, width, height = calculate_default_transform(
                src.crs, WEB_MERCATOR, src.width, src.height, *src.bounds
            )
            longest = max(width, height)
            if longest > max_dim:
                scale = max_dim / float(longest)
                transform, width, height = calculate_default_transform(
                    src.crs,
                    WEB_MERCATOR,
                    src.width,
                    src.height,
                    *src.bounds,
                    dst_width=max(1, int(width * scale)),
                    dst_height=max(1, int(height * scale)),
                )

            # Band handling
            if sensor_type == "sentinel-2":
                band_indexes = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
            else:
                band_indexes = [1, 2] if src.count >= 2 else [1, 1]

            channels = []
            for idx in band_indexes:
                destination = np.zeros((height, width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, idx),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=WEB_MERCATOR,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                )
                channels.append(destination)

            mask_dst = np.zeros((height, width), dtype=np.uint8)
            reproject(
                source=src.dataset_mask(),
                destination=mask_dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=WEB_MERCATOR,
                resampling=Resampling.nearest,
            )
            valid = (mask_dst > 0) & np.isfinite(channels[0])

            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            if sensor_type == "sentinel-2":
                for i in range(min(3, len(channels))):
                    rgba[:, :, i] = _stretch_channel(channels[i], 2.0, 98.0)
            else:
                # SAR dB false-color or grayscale
                vv_db = 10.0 * np.log10(np.clip(channels[0], 1e-6, None))
                if len(channels) >= 2:
                    vh_db = 10.0 * np.log10(np.clip(channels[1], 1e-6, None))
                    rgba[:, :, 0] = _stretch_channel(vv_db, 5.0, 95.0)
                    rgba[:, :, 1] = _stretch_channel(vh_db, 5.0, 95.0)
                    rgba[:, :, 2] = _stretch_channel(vv_db - vh_db, 5.0, 95.0)
                else:
                    stretched = _stretch_channel(vv_db, 5.0, 95.0)
                    rgba[:, :, 0] = stretched
                    rgba[:, :, 1] = stretched
                    rgba[:, :, 2] = stretched

            rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)

            target_dir = Path(output_dir) if output_dir else gtiff_file.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            png_name = f"{suffix}_{gtiff_file.stem}.png"
            png_path = target_dir / png_name
            Image.fromarray(rgba, mode="RGBA").save(str(png_path), "PNG")

            left, top = transform * (0, 0)
            right, bottom = transform * (width, height)
            wgs84 = transform_bounds(WEB_MERCATOR, "EPSG:4326", left, bottom, right, top)

            return {
                "png_path": str(png_path.resolve()),
                "png_name": png_name,
                "wgs84_bounds": [round(float(v), 8) for v in wgs84],
                "width": int(width),
                "height": int(height),
                "source_crs": str(src.crs),
            }
    except Exception as exc:
        print(f"[GIS Web Overlay Error]: {exc}")
        return None
