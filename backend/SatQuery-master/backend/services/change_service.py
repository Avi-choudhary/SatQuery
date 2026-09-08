from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.tracer import Tracer

# Robust path discovery for Bi-Temporal ChangeFormer backend
candidate_dirs = [
    Path(__file__).resolve().parents[4] / "Bi-Temporal ChangeFormer" / "SatqueryAI" / "backend",
    Path(__file__).resolve().parents[3] / "Bi-Temporal ChangeFormer" / "SatqueryAI" / "backend",
    Path(r"c:\Games\SatQuery\Bi-Temporal ChangeFormer\SatqueryAI\backend"),
]

CHANGEFORMER_BACKEND_DIR = None
for cand in candidate_dirs:
    if (cand / "services" / "change_service.py").exists():
        CHANGEFORMER_BACKEND_DIR = cand
        break

if CHANGEFORMER_BACKEND_DIR and str(CHANGEFORMER_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(CHANGEFORMER_BACKEND_DIR))

try:
    if CHANGEFORMER_BACKEND_DIR is None:
        raise FileNotFoundError("Could not find Bi-Temporal ChangeFormer backend directory with change_service.py")
    import importlib.util
    cf_cs_path = CHANGEFORMER_BACKEND_DIR / "services" / "change_service.py"
    spec = importlib.util.spec_from_file_location("cf_service_module", cf_cs_path)
    cf_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf_module)
    cf_run_inference = cf_module.run_inference
except Exception as exc:
    cf_run_inference = None
    _import_err = exc
else:
    _import_err = None


async def run_inference(
    query: str,
    file_paths: List[str],
    tracer: Tracer,
) -> Tuple[str, List[Any]]:
    """
    Invokes the Bi-Temporal Change Detective & ChangeFormerV6 specialist tool.
    Accepts two temporal rasters (before and after), aligns them to a common grid,
    computes spectral change magnitudes / deep transformer features, and returns
    a natural-language synthesis plus WGS84 GeoJSON FeatureCollection evidence.
    """
    tracer.append_log("step 2: routed to Change Detective & ChangeFormer AI specialist")

    if cf_run_inference is not None:
        try:
            return await cf_run_inference(query, file_paths, tracer)
        except Exception as exc:
            tracer.append_log(f"Change Detective execution error: {type(exc).__name__}: {exc}")
            return f"Change analysis could not be completed. Reason: {exc}", []

    tracer.append_log(f"warning: Bi-Temporal ChangeFormer pipeline unavailable: {_import_err}")
    # Fallback GeoJSON
    fallback_poly: Dict[str, Any] = {
        "type": "FeatureCollection",
        "features": []
    }
    return "Bi-Temporal ChangeFormer specialist is not available on this host.", [fallback_poly]

