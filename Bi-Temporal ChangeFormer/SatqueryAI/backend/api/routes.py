import os
import shutil
import uuid
import base64
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from core.agent import process_query
from schemas.responses import SatQueryResponse


router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]
TEMP_DIR = BASE_DIR / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Also reference the central SatQuery temp_uploads for shared presets and user uploads
CENTRAL_TEMP_DIR = BASE_DIR.parents[2] / "backend" / "SatQuery-master" / "backend" / "temp_uploads"

ALLOWED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
}


def _resolve_target_imagery(dataset_name: Optional[str] = None) -> List[str]:
    """
    Resolves target imagery paths from dataset_name, preset pairs, or recent uploads.
    Supports bi-temporal pairs when dataset_name references a multi-temporal scene.
    """
    search_dirs = [TEMP_DIR]
    if CENTRAL_TEMP_DIR.exists():
        search_dirs.append(CENTRAL_TEMP_DIR)

    if isinstance(dataset_name, str) and dataset_name.strip() and dataset_name.strip().lower() not in ["uploaded scene", "none", "null"]:
        clean = dataset_name.strip()
        clean_lower = clean.lower()

        # Check for bi-temporal preset references (Bengaluru urban corridor)
        if "bengaluru" in clean_lower and ("t1_t2" in clean_lower or "pair" in clean_lower or "urban" in clean_lower):
            for sdir in search_dirs:
                t1 = sdir / "Bengaluru_T1_Pre.png"
                t2 = sdir / "Bengaluru_T2_Post.png"
                if t1.exists() and t2.exists():
                    return [str(t1), str(t2)]

        # Check for bi-temporal preset references (Delhi multi-year)
        if "delhi" in clean_lower and ("pair" in clean_lower or "temporal" in clean_lower or "change" in clean_lower):
            for sdir in search_dirs:
                t1 = sdir / "east_delhi_2018_S2.tif"
                t2 = sdir / "delhi_20260112_S2.tif"
                if t1.exists() and t2.exists():
                    return [str(t1), str(t2)]

        # Direct file check across directories
        for sdir in search_dirs:
            cand = sdir / clean
            if cand.exists() and cand.is_file():
                return [str(cand)]

        # Matching stem check
        clean_stem = Path(clean).stem.lower()
        for sdir in search_dirs:
            for f in sdir.iterdir():
                if f.name.startswith("prev_"):
                    continue
                if f.name.lower() == clean_lower or f.stem.lower() == clean_stem:
                    if f.is_file():
                        return [str(f)]

    # Fallback to newest valid imagery in search directories
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


@router.post("/satquery", response_model=SatQueryResponse)
async def handle_satquery(
    query: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    before_file: Optional[UploadFile] = File(None),
    after_file: Optional[UploadFile] = File(None),
    dataset_name: Optional[str] = Form(None),
):
    """
    Main SatQuery endpoint supporting both single-image tasks (VQA, Grounding)
    and bi-temporal tasks (Change Detection).
    
    Accepts:
    - Multiple files via `files` (preferred for SatQuery frontend)
    - Dedicated `before_file` / `after_file` (for standalone bi-temporal tools)
    - Existing dataset or preset references via `dataset_name`
    """
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    saved_paths: List[str] = []

    try:
        # 1. Process files list from multipart
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

        # 2. Process legacy before_file and after_file if provided
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

        # 3. If no files uploaded in this request, resolve from dataset_name or fallback
        clean_dataset_name = dataset_name if isinstance(dataset_name, str) else None
        if not saved_paths:
            saved_paths = _resolve_target_imagery(clean_dataset_name)

        if not saved_paths:
            raise HTTPException(
                status_code=400,
                detail="At least one satellite image is required or must be selected via dataset_name.",
            )

        # 4. Agentic orchestration
        return await process_query(
            query=query,
            file_paths=saved_paths,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SatQuery execution error: {str(exc)}",
        )


@router.post("/satquery/json", response_model=SatQueryResponse)
async def handle_satquery_json(req: JsonQueryRequest):
    """
    JSON query endpoint for web frontends without multipart boundary requirements.
    """
    if not req.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
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

        return await process_query(
            query=req.query,
            file_paths=saved_paths,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SatQuery JSON execution error: {str(exc)}",
        )