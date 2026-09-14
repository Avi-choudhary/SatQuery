"""
AOI Satellite Scene Fetcher Orchestrator.
Main entry point connecting STAC search, windowed COG streaming,
preview rendering, and model radiometric normalization.
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from .config import DEFAULT_MAX_CLOUD_COVER, FETCHED_DIR
from .stac_search import (
    search_sentinel2,
    search_sentinel1,
    get_best_scene,
    extract_item_metadata
)
from .cog_reader import (
    fetch_aoi_raster,
    compute_bbox_area_km2,
    validate_aoi
)
from .preview_generator import generate_preview, generate_web_overlay
from .preprocessing import preprocess_for_model


def fetch_scene(
    bbox: List[float],
    sensor: str = "sentinel-2",
    max_cloud_cover: int = DEFAULT_MAX_CLOUD_COVER,
    date_range: Optional[str] = None,
    preprocess: bool = True,
    generate_png: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Fetches satellite imagery covering an arbitrary bounding box on Earth.
    
    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84 decimal degrees.
        sensor: "sentinel-2" (optical) or "sentinel-1" (SAR).
        max_cloud_cover: Maximum acceptable cloud coverage percent (Sentinel-2 only).
        date_range: Optional ISO interval, e.g. "2026-06-01T00:00:00Z/2026-09-12T00:00:00Z".
        preprocess: Whether to produce a normalized, model-ready GeoTIFF.
        generate_png: Whether to generate a contrast-enhanced PNG web preview.
        output_dir: Custom output directory (defaults to interactive_map/data/fetched).
        
    Returns:
        Dict containing execution status, paths to generated artifacts, and metadata.
    """
    t0 = time.time()
    sensor_normalized = sensor.lower().strip()
    
    # 1. Validate bounding box coordinates & area limit
    try:
        validate_aoi(bbox)
    except ValueError as ve:
        return {
            "status": "error",
            "error_type": "ValidationError",
            "message": str(ve)
        }

    area_sq_km = compute_bbox_area_km2(bbox)

    # 2. STAC Search
    t_search_start = time.time()
    if sensor_normalized in ["sentinel-1", "s1", "sar"]:
        sensor_type = "sentinel-1"
        items = search_sentinel1(bbox=bbox, date_range=date_range)
    else:
        sensor_type = "sentinel-2"
        items = search_sentinel2(
            bbox=bbox,
            max_cloud_cover=max_cloud_cover,
            date_range=date_range
        )
    t_search = time.time() - t_search_start

    if not items:
        return {
            "status": "not_found",
            "message": f"No clear {sensor_type} scenes found for bbox {bbox} in specified date window.",
            "bbox": bbox,
            "sensor": sensor_type,
            "search_latency_s": round(t_search, 2)
        }

    best_item = get_best_scene(items, sensor=sensor_type)
    if not best_item:
        return {
            "status": "not_found",
            "message": f"Could not determine suitable scene from {len(items)} candidates.",
            "sensor": sensor_type
        }

    scene_meta = extract_item_metadata(best_item, sensor=sensor_type)

    # 3. Windowed COG Pixel Streaming
    t_stream_start = time.time()
    try:
        raw_geotiff_path, fetch_meta = fetch_aoi_raster(
            item=best_item,
            bbox=bbox,
            sensor=sensor_type,
            output_dir=output_dir
        )
    except Exception as e:
        return {
            "status": "error",
            "error_type": "DownloadError",
            "message": f"Failed streaming pixels from remote COG: {e}",
            "scene_id": best_item.id,
            "sensor": sensor_type
        }
    t_stream = time.time() - t_stream_start

    # 4. Web Preview PNG Generation & Web Mercator Slippy Map Overlay
    t_preview_start = time.time()
    preview_png_path = None
    web_overlay = None
    if generate_png:
        try:
            preview_png_path = generate_preview(
                geotiff_path=raw_geotiff_path,
                sensor_type=sensor_type
            )
            web_overlay = generate_web_overlay(
                geotiff_path=raw_geotiff_path,
                sensor_type=sensor_type,
                output_dir=Path(raw_geotiff_path).parent
            )
        except Exception as e:
            print(f"[Warning] Preview generation failed: {e}")
    t_preview = time.time() - t_preview_start

    # 5. Model-Ready Normalization
    t_proc_start = time.time()
    processed_geotiff_path = None
    if preprocess:
        try:
            processed_geotiff_path = preprocess_for_model(
                raw_geotiff_path=raw_geotiff_path,
                sensor_type=sensor_type
            )
        except Exception as e:
            print(f"[Warning] Model preprocessing failed: {e}")
    t_proc = time.time() - t_proc_start

    total_latency = time.time() - t0

    bounds_wgs84 = web_overlay.get("wgs84_bounds") if web_overlay else bbox

    return {
        "status": "success",
        "scene_id": best_item.id,
        "acquisition_date": scene_meta.get("datetime"),
        "cloud_cover": scene_meta.get("cloud_cover"),
        "sensor": "Sentinel-2 (Optical)" if sensor_type == "sentinel-2" else "Sentinel-1 (SAR)",
        "bbox": bbox,
        "wgs84_bounds": bounds_wgs84,
        "area_sq_km": round(area_sq_km, 2),
        "crs": fetch_meta.get("crs"),
        "bands": fetch_meta.get("bands"),
        "pixel_dimensions": fetch_meta.get("pixel_dimensions"),
        "raw_geotiff_path": raw_geotiff_path,
        "processed_geotiff_path": processed_geotiff_path,
        "preview_png_path": preview_png_path,
        "web_overlay_path": web_overlay.get("png_path") if web_overlay else preview_png_path,
        "web_overlay_name": web_overlay.get("png_name") if web_overlay else (Path(preview_png_path).name if preview_png_path else None),
        "file_size_mb": fetch_meta.get("file_size_mb"),
        "timings": {
            "search_s": round(t_search, 2),
            "stream_s": round(t_stream, 2),
            "preview_s": round(t_preview, 2),
            "preprocess_s": round(t_proc, 2),
            "total_s": round(total_latency, 2)
        }
    }
