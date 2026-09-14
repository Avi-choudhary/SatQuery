import os
import shutil
import base64
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from schemas.responses import SatQueryResponse
from core.agent import process_query

router = APIRouter()

BACKEND_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = str(BACKEND_DIR / "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)


def _get_fallback_imagery() -> List[str]:
    """Finds the most recently uploaded or valid satellite scene for queries submitted without file payloads."""
    # 1. Look in TEMP_DIR for user-uploaded satellite images (preferring newest, excluding previews)
    if os.path.exists(TEMP_DIR):
        candidates = []
        for f in os.listdir(TEMP_DIR):
            if f.startswith("prev_"):
                continue
            if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                full_path = os.path.join(TEMP_DIR, f)
                if os.path.isfile(full_path) and os.path.getsize(full_path) > 1000:
                    candidates.append(full_path)

        if candidates:
            # Sort newest first
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return [candidates[0]]

    # 2. Check sample images in workspace
    workspace_root = Path(__file__).resolve().parents[3]
    sample_dirs = [
        workspace_root / "Model Training" / "data" / "images",
        workspace_root / "gis_extraction_logic" / "data"
    ]
    for sdir in sample_dirs:
        if sdir.exists():
            for f in sdir.glob("*.*"):
                if f.suffix.lower() in [".tif", ".tiff", ".png", ".jpg"]:
                    return [str(f)]

    # 3. Create a lightweight sample preview GeoTIFF/PNG
    sample_path = os.path.join(TEMP_DIR, "Bengaluru_Urban_Corridor_T1_T2.png")
    if not os.path.exists(sample_path):
        from PIL import Image
        img = Image.new("RGB", (256, 256), color=(40, 80, 50))
        img.save(sample_path)
    return [sample_path]


def _resolve_target_imagery(dataset_name: Optional[str] = None) -> List[str]:
    """Resolves target imagery path from dataset_name or returns the newest uploaded satellite scene."""
    if isinstance(dataset_name, str) and dataset_name.strip() and dataset_name.strip().lower() not in ["uploaded scene", "none", "null"]:
        clean = dataset_name.strip()
        clean_lower = clean.lower()

        # Check for bi-temporal preset references (Bengaluru urban corridor)
        if "bengaluru" in clean_lower and ("t1_t2" in clean_lower or "pair" in clean_lower or "urban" in clean_lower):
            t1 = os.path.join(TEMP_DIR, "Bengaluru_T1_Pre.png")
            t2 = os.path.join(TEMP_DIR, "Bengaluru_T2_Post.png")
            if os.path.exists(t1) and os.path.exists(t2):
                return [t1, t2]

        # Check for bi-temporal preset references (Delhi multi-year)
        if "delhi" in clean_lower and ("pair" in clean_lower or "temporal" in clean_lower or "change" in clean_lower):
            t1 = os.path.join(TEMP_DIR, "east_delhi_2018_S2.tif")
            t2 = os.path.join(TEMP_DIR, "delhi_20260112_S2.tif")
            if os.path.exists(t1) and os.path.exists(t2):
                return [t1, t2]

        # Direct check in TEMP_DIR
        cand = os.path.join(TEMP_DIR, clean)
        if os.path.exists(cand) and os.path.isfile(cand):
            return [cand]
        # Look for matching filename or stem in TEMP_DIR (case-insensitive)
        clean_stem = Path(clean).stem.lower()
        if os.path.exists(TEMP_DIR):
            for fname in os.listdir(TEMP_DIR):
                if fname.startswith("prev_"):
                    continue
                if fname.lower() == clean_lower or Path(fname).stem.lower() == clean_stem:
                    full = os.path.join(TEMP_DIR, fname)
                    if os.path.isfile(full):
                        return [full]

    return _get_fallback_imagery()


class JsonQueryRequest(BaseModel):
    query: str
    dataset_name: Optional[str] = None
    image_base64: Optional[str] = None


@router.post("/satquery", response_model=SatQueryResponse)
async def handle_satquery(
    query: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    before_file: Optional[UploadFile] = File(None),
    after_file: Optional[UploadFile] = File(None),
    dataset_name: Optional[str] = Form(None)
):
    saved_file_paths = []
    clean_dataset_name = dataset_name if isinstance(dataset_name, str) else None
    try:
        if files and isinstance(files, (list, tuple)):
            for file in files:
                if hasattr(file, "filename") and bool(file.filename) and hasattr(file, "file"):
                    file_location = os.path.join(TEMP_DIR, file.filename)
                    with open(file_location, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    saved_file_paths.append(file_location)

        # Also support dedicated before_file and after_file fields
        for file in [before_file, after_file]:
            if hasattr(file, "filename") and bool(file.filename) and hasattr(file, "file"):
                file_location = os.path.join(TEMP_DIR, file.filename)
                with open(file_location, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                saved_file_paths.append(file_location)

        # Trigger cached preview generation for any uploaded GeoTIFFs immediately
        for p_loc in saved_file_paths:
            if p_loc.lower().endswith((".tif", ".tiff")):
                try:
                    _generate_raster_preview(p_loc)
                except Exception:
                    pass

        # If no files in this multipart request, resolve via dataset_name or newest uploaded scene
        if not saved_file_paths:
            saved_file_paths = _resolve_target_imagery(clean_dataset_name)

        # process_query receives the query string (including any hidden [Instruction: ...] tags)
        response = await process_query(query, saved_file_paths)
        return response

    except Exception as e:
        is_hindi = "[Instruction: Respond strictly in Hindi" in query
        detail_msg = f"क्वेरी संसाधित करने में त्रुटि: {str(e)}" if is_hindi else f"Query processing error: {str(e)}"
        raise HTTPException(status_code=500, detail=detail_msg)


@router.post("/satquery/json", response_model=SatQueryResponse)
async def handle_satquery_json(req: JsonQueryRequest):
    """Convenient JSON endpoint for web frontends without requiring multipart boundary parsing."""
    try:
        saved_file_paths = []
        if req.image_base64:
            # Decode base64 to temp file
            data = req.image_base64
            if "," in data:
                data = data.split(",", 1)[1]
            b_bytes = base64.b64decode(data)
            fname = os.path.join(TEMP_DIR, f"upload_{int(os.path.getmtime(TEMP_DIR))}.png")
            with open(fname, "wb") as f:
                f.write(b_bytes)
            saved_file_paths.append(fname)
        else:
            saved_file_paths = _resolve_target_imagery(req.dataset_name)

        return await process_query(req.query, saved_file_paths)
    except Exception as e:
        is_hindi = "[Instruction: Respond strictly in Hindi" in req.query
        detail_msg = f"क्वेरी संसाधित करने में त्रुटि: {str(e)}" if is_hindi else f"Query processing error: {str(e)}"
        raise HTTPException(status_code=500, detail=detail_msg)


def _generate_raster_preview(file_path: str) -> Optional[str]:
    """Generates an RGB thumbnail PNG for a GeoTIFF or standard raster image."""
    try:
        from PIL import Image
        import numpy as np

        stem = Path(file_path).stem
        preview_filename = f"prev_{stem}.png"
        preview_path = os.path.join(TEMP_DIR, preview_filename)

        if os.path.exists(preview_path) and os.path.getsize(preview_path) > 1000:
            return preview_filename

        if file_path.lower().endswith((".tif", ".tiff")):
            import rasterio
            with rasterio.open(file_path) as src:
                factor = max(1, src.height // 600)
                h, w = max(1, src.height // factor), max(1, src.width // factor)
                if src.count >= 3:
                    # In Sentinel-2 products, Band 1 is Red (B4), Band 2 is Green (B3), Band 3 is Blue (B2)
                    b_r = src.read(1, out_shape=(h, w))
                    b_g = src.read(2, out_shape=(h, w))
                    b_b = src.read(3, out_shape=(h, w))
                else:
                    b_r = b_g = b_b = src.read(1, out_shape=(h, w))

                def norm(b):
                    valid = b[b > 0]
                    if valid.size > 0:
                        p2, p98 = np.percentile(valid, (2, 98))
                        if p98 <= p2:
                            p2, p98 = float(valid.min()), float(valid.max())
                    else:
                        p2, p98 = 0.0, 255.0
                    clipped = np.clip(b, p2, p98)
                    denom = max(1e-6, float(p98 - p2))
                    return ((clipped - p2) / denom * 255).astype(np.uint8)

                rgb = np.stack([norm(b_r), norm(b_g), norm(b_b)], axis=-1)
                im = Image.fromarray(rgb)
                im.save(preview_path, "PNG")
                return preview_filename
        else:
            im = Image.open(file_path).convert("RGB")
            im.thumbnail((800, 800))
            im.save(preview_path, "PNG")
            return preview_filename
    except Exception as e:
        print(f"[Preview Generation Warning]: {e}")
        return None


def _calculate_area_sq_km(bounds: List[float], center_lat: float) -> float:
    import math
    min_lon, min_lat, max_lon, max_lat = bounds
    lat_dist = abs(max_lat - min_lat) * 110.574
    lon_dist = abs(max_lon - min_lon) * (111.320 * math.cos(math.radians(center_lat)))
    return round(lat_dist * lon_dist, 2)


@router.post("/imagery/upload")
async def upload_imagery(
    files: List[UploadFile] = File(...),
    sensor_type: Optional[str] = Form(None)
):
    """
    Ingests single or bi-temporal satellite rasters (GeoTIFF, COG, PNG).
    Extracts geographic WGS84 bounds, center coordinates, CRS, sensor resolution,
    and returns georeferenced raster preview URLs for MapLibre map overlay.
    """
    import time
    from utils import gis_pipeline

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    saved_paths = []
    try:
        for file in files:
            if file.filename:
                file_location = os.path.join(TEMP_DIR, file.filename)
                # Avoid redundant disk writes if identical file already exists
                if not (os.path.exists(file_location) and os.path.getsize(file_location) > 0):
                    with open(file_location, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                saved_paths.append(file_location)

        if not saved_paths:
            raise HTTPException(status_code=400, detail="Failed to save uploaded imagery.")

        # Build the map overlay. For a georeferenced raster this warps to Web
        # Mercator so the PNG registers exactly; for a plain image it falls
        # back to a thumbnail that carries no geographic claim.
        info_t1 = gis_pipeline.get_geotiff_info(saved_paths[0])
        overlay_t1 = gis_pipeline.build_web_overlay(saved_paths[0], output_dir=TEMP_DIR)
        overlay_t2 = (
            gis_pipeline.build_web_overlay(saved_paths[1], output_dir=TEMP_DIR)
            if len(saved_paths) > 1
            else None
        )

        georeferenced = overlay_t1 is not None

        if georeferenced:
            prev_t1 = overlay_t1["png_name"]
            # The warped extent is the true footprint; the raw WGS84 envelope of
            # a rotated scene is not.
            bounds = overlay_t1["wgs84_bounds"]
        else:
            prev_t1 = _generate_raster_preview(saved_paths[0]) or os.path.basename(saved_paths[0])
            bounds = info_t1.get("wgs84_bounds")

        prev_t2 = None
        if len(saved_paths) > 1:
            if overlay_t2 is not None:
                prev_t2 = overlay_t2["png_name"]
            else:
                prev_t2 = _generate_raster_preview(saved_paths[1]) or os.path.basename(saved_paths[1])

        # A file with no CRS cannot be placed on a map. Say so instead of
        # dropping it over Bengaluru, which is what this used to do.
        center_lon = center_lat = None
        area_km = 0.0
        if bounds:
            center_lon = round((bounds[0] + bounds[2]) / 2, 6)
            center_lat = round((bounds[1] + bounds[3]) / 2, 6)
            area_km = _calculate_area_sq_km(bounds, center_lat)

        sensor_name = sensor_type or info_t1.get("sensor", "Sentinel-2 (Optical)")
        mode_val = "bi-temporal" if len(saved_paths) > 1 else "single"
        cache_bust = int(time.time())

        # Preview URLs are returned as origin-relative paths. An absolute
        # http://localhost:8000 URL only resolves on the machine running this
        # server, so it broke every client on the LAN and any dev-server proxy.
        return {
            "dataset_id": f"ds_{cache_bust}",
            "name": os.path.basename(saved_paths[0]),
            "sensor": sensor_name,
            "mode": mode_val,
            "georeferenced": georeferenced,
            "wgs84_bounds": bounds,
            "center": [center_lon, center_lat] if bounds else None,
            "crs": (overlay_t1 or {}).get("source_crs") or info_t1.get("crs", "unknown"),
            "resolution": (
                f"{info_t1['resolution'][0]}m GSD" if info_t1.get("resolution") else "unknown"
            ),
            "area_sq_km": area_km,
            "t1_image_url": f"/static/{prev_t1}?t={cache_bust}",
            "t2_image_url": f"/static/{prev_t2}?t={cache_bust}" if prev_t2 else None,
            "t1_filename": os.path.basename(saved_paths[0]),
            "t2_filename": os.path.basename(saved_paths[1]) if len(saved_paths) > 1 else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Imagery ingestion error: {str(e)}")


@router.get("/imagery/presets")
def get_imagery_presets():
    """Returns instant satellite demo presets with georeferenced overlays."""
    return [
        {
            "id": "delhi_temporal",
            "name": "Delhi_MultiYear_2018_2026_Pair.tif",
            "displayName": "Delhi NCR (2018 / 2026 Multi-Temporal Pair)",
            "sensor": "Optical (Sentinel-2 Multi-Temporal)",
            "mode": "bi-temporal",
            "wgs84_bounds": [77.265, 28.580, 77.365, 28.690],
            "center": [77.315, 28.635],
            "crs": "EPSG:32643 (UTM 43N)",
            "resolution": "10.0m GSD",
            "area_sq_km": 124.5,
            "t1_image_url": "/static/prev_east_delhi_2018_S2.png",
            "t2_image_url": "/static/prev_delhi_20260112_S2.png"
        },
        {
            "id": "delhi_s2",
            "name": "delhi_20260112_S2.tif",
            "displayName": "Delhi NCR (Real Sentinel-2 117MB COG)",
            "sensor": "Optical (Sentinel-2)",
            "mode": "single",
            "wgs84_bounds": [77.044156, 28.545082, 77.356719, 28.854927],
            "center": [77.200438, 28.700004],
            "crs": "EPSG:32643 (UTM 43N)",
            "resolution": "10.0m GSD",
            "area_sq_km": 1050.4,
            "t1_image_url": "/static/delhi_preview.png"
        },
        {
            "id": "bengaluru_urban_pair",
            "name": "Bengaluru_Urban_Corridor_T1_T2.tif",
            "displayName": "Bengaluru Urban Corridor (Bi-Temporal T1/T2)",
            "sensor": "Optical (Sentinel-2)",
            "mode": "bi-temporal",
            "wgs84_bounds": [77.618, 13.022, 77.652, 13.048],
            "center": [77.635, 13.035],
            "crs": "EPSG:4326 (WGS84)",
            "resolution": "10.0m GSD",
            "area_sq_km": 10.5,
            "t1_image_url": "/static/Bengaluru_T1_Pre.png",
            "t2_image_url": "/static/Bengaluru_T2_Post.png"
        },
        {
            "id": "mangalore_sar",
            "name": "Mangalore_Harbor_SAR_VV.tif",
            "displayName": "Mangalore Port (RISAT-1A / EOS-04 SAR)",
            "sensor": "SAR (RISAT-1A / EOS-04)",
            "mode": "single",
            "wgs84_bounds": [74.780, 12.850, 74.880, 12.950],
            "center": [74.830, 12.900],
            "crs": "EPSG:4326 (WGS84)",
            "resolution": "2.0m GSD (FRS Mode)",
            "area_sq_km": 118.2,
            "t1_image_url": "/static/Mangalore_SAR_VV.png"
        },
        {
            "id": "cartosat_ahmedabad",
            "name": "Ahmedabad_Cartosat3_Pan.tif",
            "displayName": "Ahmedabad City (Cartosat-3 High-Res)",
            "sensor": "Optical (Cartosat-3 PAN)",
            "mode": "single",
            "wgs84_bounds": [72.5714, 23.0225, 72.6114, 23.0625],
            "center": [72.5914, 23.0425],
            "crs": "EPSG:32643 (UTM 43N)",
            "resolution": "0.28m GSD",
            "area_sq_km": 15.6,
            "t1_image_url": "/static/Bengaluru_T1_Pre.png" 
        }
    ]



