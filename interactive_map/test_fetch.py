"""
Standalone Verification Script for Interactive Map Satellite Fetcher.
Tests Sentinel-2 (optical) and Sentinel-1 (SAR) AOI fetching, radiometric fidelity,
file sizes, preview rendering, and AOI safety constraints.
"""

import sys
import os
from pathlib import Path

# Auto-detect and re-execute in workspace .venv if system python lacks rasterio/pystac_client
try:
    import rasterio
    import pystac_client
    import numpy as np
    from PIL import Image
except ImportError:
    workspace_root = Path(__file__).resolve().parent.parent
    venv_py = workspace_root / ".venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        venv_py = workspace_root / "backend" / "SatQuery-master" / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists() and sys.executable.lower() != str(venv_py).lower():
        import subprocess
        result = subprocess.run([str(venv_py)] + sys.argv)
        sys.exit(result.returncode)
    else:
        print("[Error] Missing required packages ('rasterio', 'pystac-client'). Please activate your virtualenv (.venv).")
        sys.exit(1)

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interactive_map import fetch_scene
from interactive_map.cog_reader import compute_bbox_area_km2


def test_sentinel2():
    print("\n" + "=" * 60)
    print("TEST 1: Sentinel-2 Optical Imagery Fetch")
    print("=" * 60)
    # Delhi NCR 10km x 10km box
    bbox = [77.10, 28.55, 77.20, 28.65]
    area = compute_bbox_area_km2(bbox)
    print(f"Target AOI: {bbox} (~{area:.1f} km²)")
    
    res = fetch_scene(
        bbox=bbox,
        sensor="sentinel-2",
        max_cloud_cover=20,
        preprocess=True,
        generate_png=True
    )
    
    print(f"Fetch Status: {res.get('status')}")
    if res.get("status") != "success":
        print(f"FAILED: {res.get('message')}")
        return False
        
    print(f"  Scene ID:         {res.get('scene_id')}")
    print(f"  Acquisition Date: {res.get('acquisition_date')}")
    print(f"  Cloud Cover:      {res.get('cloud_cover')}%")
    print(f"  CRS:              {res.get('crs')}")
    print(f"  Pixel Dims:       {res.get('pixel_dimensions')} (W x H)")
    print(f"  Bands Loaded:     {res.get('bands')}")
    print(f"  Raw GeoTIFF Size: {res.get('file_size_mb')} MB")
    print(f"  Timings:          {res.get('timings')}")

    # Validate Raw GeoTIFF
    raw_path = res.get("raw_geotiff_path")
    assert os.path.exists(raw_path), "Raw GeoTIFF does not exist!"
    with rasterio.open(raw_path) as src:
        assert src.count >= 3, f"Expected at least 3 bands, got {src.count}"
        raw_arr = src.read(1)
        print(f"  Raw Band 1 (Red) dtype={raw_arr.dtype}, min={raw_arr.min()}, max={raw_arr.max()}")
        assert raw_arr.dtype == np.uint16, f"Expected uint16, got {raw_arr.dtype}"
        assert raw_arr.max() > 0, "Image contains all zeros!"

    # Validate Preprocessed GeoTIFF
    proc_path = res.get("processed_geotiff_path")
    assert os.path.exists(proc_path), "Processed GeoTIFF does not exist!"
    with rasterio.open(proc_path) as src:
        proc_arr = src.read(1)
        print(f"  Proc Band 1 dtype={proc_arr.dtype}, min={proc_arr.min():.4f}, max={proc_arr.max():.4f}")
        assert proc_arr.dtype == np.float32, f"Expected float32, got {proc_arr.dtype}"
        assert 0.0 <= proc_arr.min() <= proc_arr.max() <= 1.0, "Preprocessed values outside [0.0, 1.0]!"

    # Validate Preview PNG & Web Overlay
    png_path = res.get("preview_png_path")
    assert os.path.exists(png_path), "Preview PNG does not exist!"
    with Image.open(png_path) as im:
        print(f"  Preview PNG: mode={im.mode}, size={im.size}, file_size={os.path.getsize(png_path)/1024:.1f} KB")
        assert im.size[0] > 0 and im.size[1] > 0

    web_path = res.get("web_overlay_path")
    if web_path and os.path.exists(web_path):
        with Image.open(web_path) as im:
            print(f"  Web Mercator Overlay: mode={im.mode}, size={im.size}, bounds={res.get('wgs84_bounds')}")
            assert im.mode == "RGBA", "Web overlay must be RGBA with alpha transparency!"

    print(">>> Sentinel-2 Test PASSED!")
    return True


def test_sentinel1():
    print("\n" + "=" * 60)
    print("TEST 2: Sentinel-1 SAR Imagery Fetch")
    print("=" * 60)
    bbox = [77.10, 28.55, 77.20, 28.65]
    area = compute_bbox_area_km2(bbox)
    print(f"Target AOI: {bbox} (~{area:.1f} km²)")

    res = fetch_scene(
        bbox=bbox,
        sensor="sentinel-1",
        preprocess=True,
        generate_png=True
    )

    print(f"Fetch Status: {res.get('status')}")
    if res.get("status") != "success":
        print(f"FAILED: {res.get('message')}")
        return False

    print(f"  Scene ID:         {res.get('scene_id')}")
    print(f"  Acquisition Date: {res.get('acquisition_date')}")
    print(f"  CRS:              {res.get('crs')}")
    print(f"  Pixel Dims:       {res.get('pixel_dimensions')} (W x H)")
    print(f"  Bands Loaded:     {res.get('bands')}")
    print(f"  Raw GeoTIFF Size: {res.get('file_size_mb')} MB")
    print(f"  Timings:          {res.get('timings')}")

    # Validate Raw GeoTIFF
    raw_path = res.get("raw_geotiff_path")
    assert os.path.exists(raw_path), "Raw SAR GeoTIFF does not exist!"
    with rasterio.open(raw_path) as src:
        assert src.count >= 1, f"Expected at least 1 band, got {src.count}"
        raw_arr = src.read(1)
        print(f"  Raw SAR Band 1 dtype={raw_arr.dtype}, min={raw_arr.min()}, max={raw_arr.max()}")
        assert raw_arr.max() > 0, "SAR image contains all zeros!"

    # Validate Preprocessed SAR GeoTIFF (dB + Lee filter + normalized)
    proc_path = res.get("processed_geotiff_path")
    assert os.path.exists(proc_path), "Processed SAR GeoTIFF does not exist!"
    with rasterio.open(proc_path) as src:
        proc_arr = src.read(1)
        print(f"  Proc SAR Band 1 dtype={proc_arr.dtype}, min={proc_arr.min():.4f}, max={proc_arr.max():.4f}")
        assert proc_arr.dtype == np.float32, f"Expected float32, got {proc_arr.dtype}"
        assert 0.0 <= proc_arr.min() <= proc_arr.max() <= 1.0, "Preprocessed SAR values outside [0.0, 1.0]!"

    # Validate Preview PNG & Web Overlay
    png_path = res.get("preview_png_path")
    assert os.path.exists(png_path), "SAR Preview PNG does not exist!"
    with Image.open(png_path) as im:
        print(f"  SAR Preview PNG: mode={im.mode}, size={im.size}, file_size={os.path.getsize(png_path)/1024:.1f} KB")

    web_path = res.get("web_overlay_path")
    if web_path and os.path.exists(web_path):
        with Image.open(web_path) as im:
            print(f"  Web Mercator SAR Overlay: mode={im.mode}, size={im.size}, bounds={res.get('wgs84_bounds')}")
            assert im.mode == "RGBA", "SAR Web overlay must be RGBA with alpha transparency!"

    print(">>> Sentinel-1 Test PASSED!")
    return True


def test_safety_constraints():
    print("\n" + "=" * 60)
    print("TEST 3: Safety Constraints & Validation")
    print("=" * 60)
    
    # 1. Oversized AOI (> 2500 sq km)
    huge_bbox = [70.0, 20.0, 80.0, 30.0]
    area = compute_bbox_area_km2(huge_bbox)
    print(f"Testing oversized bbox: area = {area:,.1f} km²")
    res = fetch_scene(bbox=huge_bbox, sensor="sentinel-2")
    assert res.get("status") == "error", f"Expected error, got {res.get('status')}"
    print(f"  Correctly rejected oversized AOI: {res.get('message')}")

    # 2. Inverted coordinates
    bad_bbox = [78.0, 28.0, 77.0, 29.0]
    res_bad = fetch_scene(bbox=bad_bbox, sensor="sentinel-2")
    assert res_bad.get("status") == "error"
    print(f"  Correctly rejected inverted coords: {res_bad.get('message')}")

    print(">>> Safety Constraints Test PASSED!")
    return True


if __name__ == "__main__":
    print("Starting Interactive Map Module Automated Verification Suite...")
    s2_ok = test_sentinel2()
    s1_ok = test_sentinel1()
    safe_ok = test_safety_constraints()
    
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    print(f"Sentinel-2 Optical Fetch:  {'PASS' if s2_ok else 'FAIL'}")
    print(f"Sentinel-1 SAR Fetch:      {'PASS' if s1_ok else 'FAIL'}")
    print(f"Safety Constraints Check:  {'PASS' if safe_ok else 'FAIL'}")
    print("=" * 60)
    
    if s2_ok and s1_ok and safe_ok:
        print("ALL TESTS PASSED SUCCESSFULLY! Phase 1 is verified and ready.")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)
