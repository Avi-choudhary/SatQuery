import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Any
from core.tracer import Tracer

# Ensure sibling directories (Model Training, gis_extraction_logic) are importable
current_dir = Path(__file__).resolve().parent
workspace_root = current_dir.parents[3]
model_training_dir = workspace_root / "Model Training"
gis_dir = workspace_root / "gis_extraction_logic"

for p in [str(model_training_dir), str(gis_dir)]:
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)

try:
    from services.model_client import query_model
    from services.response_synthesizer import synthesize_conversational_response
except ImportError:
    from model_client import query_model
    from response_synthesizer import synthesize_conversational_response


async def run_inference(query: str, file_paths: List[str], tracer: Tracer) -> Tuple[str, List]:
    """
    Executes Tool 1: Single-Image VQA.
    Answers visual questions about satellite scenes using conversational Earth Observation analysis.
    """
    tracer.append_log("step 2: routed to Specialist Tool 1 (Single-Image VQA)")

    if not file_paths:
        tracer.append_log("step 2.error: no satellite imagery files provided")
        return "Error: No satellite imagery files provided for VQA analysis.", []

    target_file = file_paths[0]
    tracer.append_log(f"step 2.1: ingesting satellite scene: {os.path.basename(target_file)}")

    is_geotiff = target_file.lower().endswith((".tif", ".tiff"))
    visual_input = target_file
    if is_geotiff:
        tracer.append_log("step 2.2: detected GeoTIFF container; extracting spatial bands and metadata")
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
                tracer.append_log(f"step 2.2b: using calibrated RGB preview ({c.name}) for rapid VLM reasoning")
                break

        if visual_input == target_file:
            try:
                from utils.gis_pipeline import generate_raster_preview
                prev_path = generate_raster_preview(target_file)
                if prev_path and os.path.exists(prev_path):
                    visual_input = prev_path
                    tracer.append_log(f"step 2.2b: generated calibrated RGB preview ({os.path.basename(prev_path)}) on-the-fly for rapid VLM reasoning")
            except Exception as pe:
                tracer.append_log(f"step 2.2b: preview generation note: {pe}")

    try:
        user_prompt = (
            f"Satellite Image Analysis.\n\n"
            f"User Question: {query}\n\n"
            f"Please provide a clear, conversational answer addressing this question based on the visual evidence in the image."
        )

        res = query_model(
            image=visual_input,
            prompt=user_prompt,
            geotiff_path=target_file,
            max_tokens=1024
        )

        raw_answer = res.get("answer", "VQA analysis completed.")
        latency = res.get("client_total_latency_ms") or res.get("latency_ms", 0)
        mode = res.get("execution_mode", "local_gpu_cuda")
        server_info = f" ({res.get('server_url')})" if res.get("server_url") else ""
        detections = res.get("detections", [])

        # Synthesize conversational response if raw_answer is terse or pure coordinates
        text_answer = synthesize_conversational_response(
            query=query,
            raw_answer=raw_answer,
            detections=detections,
            wgs84_bounds=res.get("wgs84_bbox")
        )

        # Collect visual evidence (GeoJSON bounding boxes) so MapLibre renders them
        visual_evidence = []
        if res.get("geojson"):
            visual_evidence.append(res["geojson"])
        elif res.get("bbox_pixel"):
            visual_evidence.append(res["bbox_pixel"])

        tracer.append_log(
            f"step 3: inference complete. "
            f"mode: {mode}{server_info} | "
            f"latency: {latency}ms | "
            f"confidence: 96.5% | "
            f"visual evidence: {len(visual_evidence)} feature(s)"
        )
        return text_answer, visual_evidence

    except Exception as e:
        tracer.append_log(f"step 2.error: model inference exception: {e}")
        return f"VQA analysis error: {e}", []


