import os
import shutil
import uuid
import base64
import math
import time
from pathlib import Path
from typing import List, Optional

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth.dependencies import get_current_user, get_optional_current_user
from core.agent import process_query
from db.session import get_db
from models.analysis_run import AnalysisRun
from models.conversation import Conversation
from models.message import Message
from models.user import User
from schemas.responses import SatQueryResponse
from utils import gis_pipeline


router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]
TEMP_DIR = BASE_DIR / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_SEARCH_DIRS = [
    TEMP_DIR,
    BASE_DIR.parents[2] / "backend" / "temp_uploads",
    BASE_DIR.parents[3] / "backend" / "temp_uploads",
    Path("/Users/divyatewari/Desktop/SatqueryAI/backend/temp_uploads"),
]

ALLOWED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
}


def _resolve_target_imagery(dataset_name: Optional[str] = None) -> List[str]:
    search_dirs = [d for d in CANDIDATE_SEARCH_DIRS if d.exists()]

    if isinstance(dataset_name, str) and dataset_name.strip() and dataset_name.strip().lower() not in ["uploaded scene", "none", "null"]:
        clean = dataset_name.strip()
        clean_lower = clean.lower()

        if "bengaluru" in clean_lower and ("t1_t2" in clean_lower or "pair" in clean_lower or "urban" in clean_lower):
            for sdir in search_dirs:
                t1 = sdir / "Bengaluru_T1_Pre.png"
                t2 = sdir / "Bengaluru_T2_Post.png"
                if t1.exists() and t2.exists():
                    return [str(t1), str(t2)]

        if "delhi" in clean_lower and ("pair" in clean_lower or "temporal" in clean_lower or "change" in clean_lower):
            for sdir in search_dirs:
                t1 = sdir / "east_delhi_2018_S2.tif"
                t2 = sdir / "delhi_20260112_S2.tif"
                if t1.exists() and t2.exists():
                    return [str(t1), str(t2)]

        for sdir in search_dirs:
            cand = sdir / clean
            if cand.exists() and cand.is_file():
                return [str(cand)]

        clean_stem = Path(clean).stem.lower()
        for sdir in search_dirs:
            for f in sdir.iterdir():
                if f.name.startswith("prev_"):
                    continue
                if f.name.lower() == clean_lower or f.stem.lower() == clean_stem:
                    if f.is_file():
                        return [str(f)]

    for sdir in search_dirs:
        candidates = []
        for f in sdir.iterdir():
            if f.name.startswith("prev_") or f.name.startswith("."):
                continue
            if f.suffix.lower() in ALLOWED_EXTENSIONS and f.is_file() and f.stat().st_size > 1000:
                candidates.append(f)
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return [str(candidates[0])]

    return []


class JsonQueryRequest(BaseModel):
    query: str
    dataset_name: Optional[str] = None
    image_base64: Optional[str] = None
    conversation_id: Optional[str] = None


@router.post("/satquery", response_model=SatQueryResponse)
async def handle_satquery(
    query: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    before_file: Optional[UploadFile] = File(None),
    after_file: Optional[UploadFile] = File(None),
    dataset_name: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    # 1. Resolve or create user-owned conversation if authenticated
    conv = None
    if current_user is not None:
        if conversation_id and conversation_id.strip():
            conv = (
                db.query(Conversation)
                .filter(Conversation.id == conversation_id.strip(), Conversation.user_id == current_user.id)
                .first()
            )
            if not conv:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found or access denied.",
                )
        else:
            title_snippet = query.strip().split("\n")[0][:80]
            conv = Conversation(
                user_id=current_user.id,
                title=title_snippet or "New Conversation",
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)

        # 2. Persist user message
        user_msg = Message(
            conversation_id=conv.id,
            role="user",
            content=query.strip(),
        )
        db.add(user_msg)
        db.commit()
    elif conversation_id and conversation_id.strip():
        # Unauthenticated request trying to access a conversation
        raise HTTPException(
            status_code=404,
            detail="Conversation not found or access denied.",
        )

    saved_paths: List[str] = []

    try:
        if files and isinstance(files, (list, tuple)):
            for upload in files:
                if hasattr(upload, "filename") and bool(upload.filename) and hasattr(upload, "file"):
                    suffix = Path(upload.filename).suffix.lower()
                    if suffix not in ALLOWED_EXTENSIONS:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unsupported file type: {suffix}. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
                        )
                    dest = TEMP_DIR / f"{uuid.uuid4().hex}_{upload.filename}"
                    with dest.open("wb") as out:
                        shutil.copyfileobj(upload.file, out)
                    saved_paths.append(str(dest))

        for upload in [before_file, after_file]:
            if hasattr(upload, "filename") and bool(upload.filename) and hasattr(upload, "file"):
                suffix = Path(upload.filename).suffix.lower()
                if suffix not in ALLOWED_EXTENSIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {suffix}. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
                    )
                dest = TEMP_DIR / f"{uuid.uuid4().hex}_{upload.filename}"
                with dest.open("wb") as out:
                    shutil.copyfileobj(upload.file, out)
                saved_paths.append(str(dest))

        clean_dataset_name = dataset_name if isinstance(dataset_name, str) else None
        if not saved_paths:
            saved_paths = _resolve_target_imagery(clean_dataset_name)

        if not saved_paths:
            raise HTTPException(
                status_code=400,
                detail="At least one satellite image is required or must be selected via dataset_name.",
            )

        res = await process_query(
            query=query,
            file_paths=saved_paths,
        )

        # 3. Persist assistant message & run metadata if authenticated
        if current_user is not None and conv is not None:
            asst_msg = Message(
                conversation_id=conv.id,
                role="assistant",
                content=res.text_answer,
                visual_evidence=res.visual_evidence,
                execution_trace=res.execution_trace,
            )
            db.add(asst_msg)

            b_file = Path(saved_paths[0]).name if len(saved_paths) > 0 else None
            a_file = Path(saved_paths[1]).name if len(saved_paths) > 1 else None
            analysis_ref = None
            if isinstance(res.execution_trace, dict):
                analysis_ref = res.execution_trace.get("output_dir") or res.execution_trace.get("analysis_reference")

            analysis_run = AnalysisRun(
                conversation_id=conv.id,
                user_id=current_user.id,
                query=query.strip(),
                before_filename=b_file,
                after_filename=a_file,
                result_summary=res.text_answer[:400],
                analysis_reference=str(analysis_ref) if analysis_ref else None,
                artifacts={"visual_evidence_count": len(res.visual_evidence)},
            )
            db.add(analysis_run)

            conv.updated_at = datetime.now(timezone.utc)
            db.commit()

            res.conversation_id = conv.id

        return res

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SatQuery execution error: {str(exc)}",
        )


@router.post("/satquery/json", response_model=SatQueryResponse)
async def handle_satquery_json(
    req: JsonQueryRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    if not req.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    # 1. Resolve or create user-owned conversation if authenticated
    conv = None
    if current_user is not None:
        if req.conversation_id and req.conversation_id.strip():
            conv = (
                db.query(Conversation)
                .filter(Conversation.id == req.conversation_id.strip(), Conversation.user_id == current_user.id)
                .first()
            )
            if not conv:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found or access denied.",
                )
        else:
            title_snippet = req.query.strip().split("\n")[0][:80]
            conv = Conversation(
                user_id=current_user.id,
                title=title_snippet or "New Conversation",
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)

        # 2. Persist user message
        user_msg = Message(
            conversation_id=conv.id,
            role="user",
            content=req.query.strip(),
        )
        db.add(user_msg)
        db.commit()
    elif req.conversation_id and req.conversation_id.strip():
        raise HTTPException(
            status_code=404,
            detail="Conversation not found or access denied.",
        )

    saved_paths: List[str] = []

    try:
        if req.image_base64:
            data = req.image_base64
            if "," in data:
                data = data.split(",", 1)[1]
            b_bytes = base64.b64decode(data)
            fname = TEMP_DIR / f"upload_{uuid.uuid4().hex}.png"
            with fname.open("wb") as f:
                f.write(b_bytes)
            saved_paths.append(str(fname))
        else:
            saved_paths = _resolve_target_imagery(req.dataset_name)

        if not saved_paths:
            raise HTTPException(
                status_code=400,
                detail="At least one satellite image is required or must be selected via dataset_name.",
            )

        res = await process_query(
            query=req.query,
            file_paths=saved_paths,
        )

        # 3. Persist assistant message & run metadata if authenticated
        if current_user is not None and conv is not None:
            asst_msg = Message(
                conversation_id=conv.id,
                role="assistant",
                content=res.text_answer,
                visual_evidence=res.visual_evidence,
                execution_trace=res.execution_trace,
            )
            db.add(asst_msg)

            b_file = Path(saved_paths[0]).name if len(saved_paths) > 0 else None
            a_file = Path(saved_paths[1]).name if len(saved_paths) > 1 else None
            analysis_ref = None
            if isinstance(res.execution_trace, dict):
                analysis_ref = res.execution_trace.get("output_dir") or res.execution_trace.get("analysis_reference")

            analysis_run = AnalysisRun(
                conversation_id=conv.id,
                user_id=current_user.id,
                query=req.query.strip(),
                before_filename=b_file,
                after_filename=a_file,
                result_summary=res.text_answer[:400],
                analysis_reference=str(analysis_ref) if analysis_ref else None,
                artifacts={"visual_evidence_count": len(res.visual_evidence)},
            )
            db.add(analysis_run)

            conv.updated_at = datetime.now(timezone.utc)
            db.commit()

            res.conversation_id = conv.id

        return res

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SatQuery JSON execution error: {str(exc)}",
        )


def _generate_raster_preview(file_path: Path) -> Optional[str]:
    try:
        stem = file_path.stem
        prev_name = f"prev_{stem}.png"
        prev_path = TEMP_DIR / prev_name
        if prev_path.exists() and prev_path.stat().st_size > 1000:
            return prev_name
        if file_path.suffix.lower() in {".tif", ".tiff"}:
            import rasterio
            import numpy as np
            from PIL import Image
            with rasterio.open(file_path) as src:
                factor = max(1, max(src.height, src.width) // 512)
                h, w = max(1, src.height // factor), max(1, src.width // factor)
                if src.count >= 3:
                    r = src.read(1, out_shape=(h, w))
                    g = src.read(2, out_shape=(h, w))
                    b = src.read(3, out_shape=(h, w))
                else:
                    r = g = b = src.read(1, out_shape=(h, w))
                rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
                for c in range(3):
                    lo, hi = np.percentile(rgb[..., c], [2, 98])
                    if hi > lo:
                        rgb[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo), 0, 1)
                img = Image.fromarray((rgb * 255).astype(np.uint8))
                img.save(prev_path)
                return prev_name
        elif file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            from PIL import Image
            img = Image.open(file_path)
            img.thumbnail((512, 512))
            img.save(prev_path, "PNG")
            return prev_name
    except Exception:
        pass
    return None


def _calculate_area_sq_km(bounds: List[float], center_lat: Optional[float] = None) -> float:
    if not bounds or len(bounds) != 4:
        return 0.0
    lat = center_lat if center_lat is not None else (bounds[1] + bounds[3]) / 2.0
    lat_dist = abs(bounds[3] - bounds[1]) * 110.574
    lon_dist = abs(bounds[2] - bounds[0]) * (111.320 * math.cos(math.radians(lat)))
    return round(lat_dist * lon_dist, 2)


@router.post("/imagery/upload")
async def upload_imagery(
    files: List[UploadFile] = File(...),
    sensor_type: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    saved_paths: List[str] = []
    for file in files:
        if file.filename:
            dest = TEMP_DIR / file.filename
            if not (dest.exists() and dest.stat().st_size > 0):
                with dest.open("wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            saved_paths.append(str(dest))
    if not saved_paths:
        raise HTTPException(status_code=400, detail="Failed to save uploaded imagery.")

    try:
        # Build the map overlay. For a georeferenced raster this warps to Web
        # Mercator so the PNG registers exactly; for a plain image it falls
        # back to a thumbnail that carries no geographic claim.
        info_t1 = gis_pipeline.get_geotiff_info(saved_paths[0])
        overlay_t1 = gis_pipeline.build_web_overlay(saved_paths[0], output_dir=str(TEMP_DIR))
        overlay_t2 = (
            gis_pipeline.build_web_overlay(saved_paths[1], output_dir=str(TEMP_DIR))
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
            prev_t1 = _generate_raster_preview(Path(saved_paths[0])) or os.path.basename(saved_paths[0])
            bounds = info_t1.get("wgs84_bounds")

        prev_t2 = None
        if len(saved_paths) > 1:
            if overlay_t2 is not None:
                prev_t2 = overlay_t2["png_name"]
            else:
                prev_t2 = _generate_raster_preview(Path(saved_paths[1])) or os.path.basename(saved_paths[1])

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
            "t2_image_url": "/static/prev_delhi_20260112_S2.png",
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
            "t1_image_url": "/static/prev_delhi_20260112_S2.png",
        },
        {
            "id": "bengaluru_urban_pair",
            "name": "bengaluru_urban_2022_2024.tif",
            "displayName": "Bengaluru Urban (2022 / 2024 Expansion)",
            "sensor": "Sentinel-2 (Optical Multi-Temporal)",
            "mode": "bi-temporal",
            "wgs84_bounds": [77.618, 13.022, 77.652, 13.048],
            "center": [77.635, 13.035],
            "crs": "EPSG:4326 (WGS84)",
            "resolution": "10.0m GSD",
            "area_sq_km": 10.5,
            "t1_image_url": "/static/Bengaluru_T1_Pre.png",
            "t2_image_url": "/static/Bengaluru_T2_Post.png",
        },
        {
            "id": "mangalore_sar",
            "name": "mangalore_sar_surface_water.tif",
            "displayName": "Mangalore Coast (Sentinel-1 SAR C-Band)",
            "sensor": "Sentinel-1 GRD (SAR)",
            "mode": "single",
            "wgs84_bounds": [74.792, 12.825, 74.912, 12.965],
            "center": [74.852, 12.895],
            "crs": "EPSG:4326 (WGS84)",
            "resolution": "10.0m GSD",
            "area_sq_km": 118.2,
            "t1_image_url": "/static/Mangalore_SAR_VV.png",
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
            "t1_image_url": "/static/Bengaluru_T1_Pre.png",
        },
    ]

