"""
Interactive Map STAC + COG API Router.
Provides endpoints for querying AWS Open Data STAC catalogs, streaming windowed
Cloud Optimized GeoTIFFs, and serving Web Mercator overlays to MapLibre GL.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Dynamically locate workspace root containing interactive_map
current_file = Path(__file__).resolve()
BACKEND_DIR = current_file.parent.parent
WORKSPACE_ROOT = None
for parent in current_file.parents:
    if (parent / "interactive_map").exists():
        WORKSPACE_ROOT = parent
        break

if WORKSPACE_ROOT and str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from interactive_map import fetch_scene
    from interactive_map.config import STAC_API_URL, MAX_AOI_AREA_SQ_KM
    from interactive_map.cog_reader import compute_bbox_area_km2
    MODULE_AVAILABLE = True
except Exception as e:
    MODULE_AVAILABLE = False
    MODULE_ERROR = str(e)

router = APIRouter()

TEMP_DIR = Path(BACKEND_DIR / "temp_uploads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class FetchAOIRequest(BaseModel):
    bbox: List[float] = Field(
        ...,
        description="[min_lon, min_lat, max_lon, max_lat] in WGS84 decimal degrees",
        min_items=4,
        max_items=4
    )
    sensor: str = Field(
        "sentinel-2",
        description="'sentinel-2' (optical) or 'sentinel-1' (SAR radar)"
    )
    max_cloud_cover: int = Field(
        20,
        description="Maximum acceptable cloud coverage (0-100) for optical scenes",
        ge=0,
        le=100
    )
    date_range: Optional[str] = Field(
        None,
        description="Optional ISO 8601 date interval, e.g. '2026-06-01T00:00:00Z/2026-09-13T00:00:00Z'"
    )


@router.get("/health")
def interactive_map_health():
    """Reports status of the STAC + COG satellite fetch engine."""
    return {
        "status": "online" if MODULE_AVAILABLE else "degraded",
        "module_loaded": MODULE_AVAILABLE,
        "error": MODULE_ERROR if not MODULE_AVAILABLE else None,
        "stac_endpoint": STAC_API_URL if MODULE_AVAILABLE else None,
        "max_aoi_cap_sq_km": MAX_AOI_AREA_SQ_KM if MODULE_AVAILABLE else 2500.0,
        "supported_sensors": ["sentinel-2", "sentinel-1"]
    }


@router.post("/fetch")
async def fetch_satellite_aoi(req: FetchAOIRequest):
    """
    Streams genuine Sentinel-1 or Sentinel-2 satellite imagery for any bounding box.
    Returns metadata formatted for the SatQuery frontend's SceneOverlay and SceneDataset.
    """
    if not MODULE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Interactive map streaming engine unavailable: {MODULE_ERROR}"
        )

    min_lon, min_lat, max_lon, max_lat = req.bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid bounding box: min coordinates must be strictly less than max. Got {req.bbox}"
        )

    area_sq_km = compute_bbox_area_km2(req.bbox)
    if area_sq_km > MAX_AOI_AREA_SQ_KM:
        raise HTTPException(
            status_code=400,
            detail=f"Requested area ({area_sq_km:.1f} km²) exceeds maximum allowed cap of {MAX_AOI_AREA_SQ_KM} km²."
        )

    # Call Phase 1 engine with output directory set to backend/temp_uploads/
    try:
        res = fetch_scene(
            bbox=req.bbox,
            sensor=req.sensor,
            max_cloud_cover=req.max_cloud_cover,
            date_range=req.date_range,
            preprocess=True,
            generate_png=True,
            output_dir=TEMP_DIR
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Remote STAC/COG streaming failure: {str(e)}"
        )

    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message", "Imagery fetch error"))
    if res.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=res.get("message", "No matching satellite scenes found"))

    raw_path = Path(res["raw_geotiff_path"])
    t1_filename = raw_path.name
    
    # Use Web Mercator overlay for map if available, otherwise standard preview
    web_overlay_name = res.get("web_overlay_name")
    if not web_overlay_name and res.get("preview_png_path"):
        web_overlay_name = Path(res["preview_png_path"]).name

    wgs84_bounds = res.get("wgs84_bounds") or req.bbox
    center_lon = round((wgs84_bounds[0] + wgs84_bounds[2]) / 2.0, 6)
    center_lat = round((wgs84_bounds[1] + wgs84_bounds[3]) / 2.0, 6)

    is_optical = "sentinel-2" in req.sensor.lower()
    sensor_display = "Optical (Sentinel-2)" if is_optical else "SAR (Sentinel-1)"

    suggested_queries = [
        "What is the predominant land use and vegetative health in this scene?",
        "Identify urban infrastructure, roadways, and built-up structures.",
        "Analyze water bodies, agricultural fields, and natural canopy coverage."
    ] if is_optical else [
        "Identify open water bodies, rivers, and moisture contours in this SAR scene.",
        "Analyze radar backscatter roughness across built-up urban zones.",
        "Detect terrain changes, coastal structures, and surface characteristics."
    ]

    return {
        "status": "success",
        "dataset_id": f"aoi_{res.get('scene_id', 'scene')}",
        "name": t1_filename,
        "scene_id": res.get("scene_id"),
        "sensor": sensor_display,
        "mode": "single",
        "georeferenced": True,
        "wgs84_bounds": wgs84_bounds,
        "center": [center_lon, center_lat],
        "crs": res.get("crs", "EPSG:4326"),
        "resolution": "10.0m GSD",
        "area_sq_km": res.get("area_sq_km", round(area_sq_km, 2)),
        "t1_image_url": f"/static/{web_overlay_name}",
        "t1_filename": t1_filename,
        "acquisition_date": res.get("acquisition_date"),
        "cloud_cover": res.get("cloud_cover"),
        "file_size_mb": res.get("file_size_mb", 0.0),
        "bands": res.get("bands", []),
        "timings": res.get("timings", {}),
        "suggested_queries": suggested_queries
    }
