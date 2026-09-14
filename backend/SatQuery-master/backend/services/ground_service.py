import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Any
from core.tracer import Tracer

# Ensure sibling directories (Model Training, gis_extraction_logic) are importable
current_dir = Path(__file__).resolve().parent
# Traverse up to workspace root: ground_service.py -> services -> backend -> SatQuery-master -> workspace
workspace_root = current_dir.parents[3]
model_training_dir = workspace_root / "Model Training"
gis_dir = workspace_root / "gis_extraction_logic"

for p in [str(model_training_dir), str(gis_dir)]:
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)

try:
    from services.model_client import query_model
except ImportError:
    from model_client import query_model


async def run_inference(query: str, file_paths: List[str], tracer: Tracer) -> Tuple[str, List]:
    """
    Executes Tool 2: Visual Grounding.
    Accepts natural-language query and image/GeoTIFF path(s).
    Localizes query-relevant entities and outputs text answer + WGS84 GeoJSON bounding boxes.
    Supports both local GPU execution and remote GPU server execution over HTTPS.
    """
    tracer.append_log("step 2: routed to Specialist Tool 2 (Visual Grounding)")

    if not file_paths:
        tracer.append_log("step 2.error: no satellite imagery files provided")
        return "Error: No satellite imagery files provided for visual grounding.", []

    target_file = file_paths[0]
    tracer.append_log(f"step 2.1: ingesting source satellite file: {os.path.basename(target_file)}")

    is_geotiff = target_file.lower().endswith((".tif", ".tiff"))
    visual_input = target_file
    if is_geotiff:
        tracer.append_log("step 2.2: detected GeoTIFF container; extracting spatial CRS, transform and true-color bands")
        stem = Path(target_file).stem
        parent = Path(target_file).parent
        candidates = [
            parent / f"prev_{stem}.png",
            parent / f"{stem}.png",
            parent.parent / "previews" / f"{stem}.png"
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 500:
                visual_input = str(c)
                tracer.append_log(f"step 2.2b: using calibrated RGB preview ({c.name}) for rapid visual grounding")
                break

        if visual_input == target_file:
            try:
                from utils.gis_pipeline import generate_raster_preview
                prev_path = generate_raster_preview(target_file)
                if prev_path and os.path.exists(prev_path):
                    visual_input = prev_path
                    tracer.append_log(f"step 2.2b: generated calibrated RGB preview ({os.path.basename(prev_path)}) on-the-fly for rapid visual grounding")
            except Exception as pe:
                tracer.append_log(f"step 2.2b: preview generation note: {pe}")

    try:
        res = query_model(
            image=visual_input,
            prompt=f"Analyze this Sentinel-2 satellite image.\n\nQuestion: {query}",
            geotiff_path=target_file
        )

        raw_answer = res.get("answer", "Visual grounding completed.")
        latency = res.get("client_total_latency_ms") or res.get("latency_ms", 0)
        mode = res.get("execution_mode", "local_gpu_cuda")
        server_info = f" ({res.get('server_url')})" if res.get("server_url") else ""
        detections = res.get("detections", [])

        try:
            from services.response_synthesizer import synthesize_conversational_response
        except ImportError:
            from response_synthesizer import synthesize_conversational_response

        text_answer = synthesize_conversational_response(
            query=query,
            raw_answer=raw_answer,
            scene_landcover=res.get("scene_landcover"),
            detections=detections,
            wgs84_bounds=res.get("wgs84_bbox")
        )

        visual_evidence = []
        if res.get("geojson"):
            visual_evidence.append(res["geojson"])
        elif res.get("bbox_pixel"):
            visual_evidence.append(res["bbox_pixel"])

        trace_msg = (
            f"step 3: inference complete. "
            f"mode: {mode}{server_info} | "
            f"latency: {latency}ms | "
            f"detected regions: {len(detections)} | "
            f"CRS: {'EPSG:4326' if is_geotiff else 'relative'}"
        )
        tracer.append_log(trace_msg)
        return text_answer, visual_evidence

    except Exception as e:
        tracer.append_log(f"step 2.error: model inference exception: {e}")
        return f"Grounding error: {e}", []


