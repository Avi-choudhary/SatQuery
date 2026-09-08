import json
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds

from config import (
    PATCH_SIZE,
    PATCH_STRIDE,
    PATCHES_DIR,
    PREVIEWS_DIR,
    METADATA_DIR,
    WGS84_CRS,
)


def generate_patch_windows(height, width, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    """Work out a grid of 256x256 boxes covering the whole image."""
    windows = []
    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            windows.append(Window(col, row, patch_size, patch_size))
    return windows


def patchify(array, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    """Actually cut the array into those boxes."""
    _, height, width = array.shape
    windows = generate_patch_windows(height, width, patch_size, stride)
    print(f"Total patches generated: {len(windows)}")
    patches = []
    for w in windows:
        patch = array[
            :,
            int(w.row_off) : int(w.row_off + w.height),
            int(w.col_off) : int(w.col_off + w.width),
        ]
        patches.append((patch, w))
    return patches


def generate_preview_image(patch, sensor, output_path):
    """Create an 8-bit web-friendly RGB or Grayscale PNG preview for browsers."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sensor == "S2":
        # Sentinel-2: Bands are [B4/Red, B3/Green, B2/Blue, B8/NIR]
        if patch.shape[0] >= 3:
            # Extract Red, Green, Blue bands
            rgb = patch[0:3, :, :].copy()
            # Replace NaNs or Infs
            rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
            # Transpose to (height, width, 3)
            rgb_hwc = np.transpose(rgb, (1, 2, 0))

            # 2% to 98% percentile linear stretch for clear, vibrant imagery
            p2 = np.percentile(rgb_hwc, 2)
            p98 = np.percentile(rgb_hwc, 98)
            if p98 > p2:
                stretched = np.clip((rgb_hwc - p2) / (p98 - p2), 0.0, 1.0)
            else:
                stretched = np.clip(rgb_hwc, 0.0, 1.0)

            rgb_uint8 = (stretched * 255.0).astype(np.uint8)
            img = Image.fromarray(rgb_uint8, mode="RGB")
        else:
            band = np.nan_to_num(patch[0], nan=0.0)
            p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
            stretched = np.clip((band - p2) / (p98 - p2) if p98 > p2 else band, 0.0, 1.0)
            img = Image.fromarray((stretched * 255.0).astype(np.uint8), mode="L")
    else:
        # Sentinel-1 SAR (e.g., VV, VH in dB)
        vv = patch[0, :, :].copy()
        vv_clean = np.nan_to_num(vv, nan=-25.0, posinf=0.0, neginf=-30.0)
        p2, p98 = np.percentile(vv_clean, 2), np.percentile(vv_clean, 98)
        if p98 > p2:
            norm = np.clip((vv_clean - p2) / (p98 - p2), 0.0, 1.0)
        else:
            norm = np.clip((vv_clean + 25.0) / 25.0, 0.0, 1.0)
        sar_uint8 = (norm * 255.0).astype(np.uint8)
        img = Image.fromarray(sar_uint8, mode="L")

    img.save(output_path, format="PNG")
    return output_path


def save_patch(patch, window, profile, region, date, sensor, patch_id):
    """Save one small tile as a GeoTIFF + web PNG preview + enriched JSON metadata."""
    name = f"{region}_{date}_{sensor}_p{patch_id:04d}"
    tif_path = PATCHES_DIR / f"{name}.tif"
    png_path = PREVIEWS_DIR / f"{name}.png"
    json_path = METADATA_DIR / f"{name}.json"

    # 1. Save GeoTIFF
    patch_transform = rasterio.windows.transform(window, profile["transform"])
    patch_profile = profile.copy()
    patch_profile.update(
        height=patch.shape[-2],
        width=patch.shape[-1],
        transform=patch_transform,
        count=patch.shape[0],
    )

    tif_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif_path, "w", **patch_profile) as dst:
        dst.write(patch)

    # 2. Save Web PNG Preview
    generate_preview_image(patch, sensor, png_path)

    # 3. Compute Coordinates in Projected and WGS84 CRS
    native_bounds = rasterio.windows.bounds(window, profile["transform"])
    src_crs = profile["crs"]

    try:
        if str(src_crs).upper() not in ["EPSG:4326", "WGS 84", "OGC:CRS84"]:
            wgs84_bounds = list(transform_bounds(src_crs, WGS84_CRS, *native_bounds))
        else:
            wgs84_bounds = list(native_bounds)
    except Exception:
        wgs84_bounds = list(native_bounds)

    min_lon = round(float(wgs84_bounds[0]), 6)
    min_lat = round(float(wgs84_bounds[1]), 6)
    max_lon = round(float(wgs84_bounds[2]), 6)
    max_lat = round(float(wgs84_bounds[3]), 6)

    center_lat = round((min_lat + max_lat) / 2.0, 6)
    center_lon = round((min_lon + max_lon) / 2.0, 6)

    geojson_poly = {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat]
        ]]
    }

    # 4. Save Enriched JSON Label Card
    metadata = {
        "patch_id": patch_id,
        "name": name,
        "region": region,
        "date": date,
        "sensor": sensor,
        "crs": str(profile["crs"]),
        "projected_bounds": [round(float(b), 2) for b in native_bounds],
        "bounding_box": [min_lon, min_lat, max_lon, max_lat],
        "center": {
            "lat": center_lat,
            "lon": center_lon
        },
        "shape": list(patch.shape),
        "geojson": geojson_poly,
        "tif_path": str(tif_path),
        "png_path": str(png_path),
        "png_url": f"/api/patches/{name}/preview"
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return tif_path, json_path


def tile_raster(array, profile, region, date, sensor):
    """Run the whole tiling process for one full aligned image."""
    patches = patchify(array, PATCH_SIZE, PATCH_STRIDE)
    results = []
    for patch_id, (patch, window) in enumerate(patches):
        tif_path, json_path = save_patch(patch, window, profile, region, date, sensor, patch_id)
        results.append({"tif_path": str(tif_path), "json_path": str(json_path)})
    return results