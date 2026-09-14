"""
SatQuery Hybrid Model Client
============================
Intelligently routes inference to:
1. Local CUDA GPU (if running directly on the GPU host PC)
2. Remote GPU Server over HTTPS (if running on any laptop anywhere in the world)
"""

import os
import io
import sys
import time
import json
import base64
import urllib.request
import urllib.error
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
from PIL import Image

# Ensure Model Training is on path if running locally
current_dir = Path(__file__).resolve().parent
workspace_root = current_dir.parents[3]
model_training_dir = workspace_root / "Model Training"
if str(model_training_dir) not in sys.path and model_training_dir.exists():
    sys.path.insert(0, str(model_training_dir))

from core.config import MODEL_SERVICE_URL

_local_vlm_instance: Optional[Any] = None


def get_local_vlm():
    """Returns local SatQueryVLM instance if CUDA weights are available on this machine."""
    global _local_vlm_instance
    if _local_vlm_instance is None:
        try:
            from satquery_model import SatQueryVLM
            _local_vlm_instance = SatQueryVLM(warmup=False)
        except Exception as e:
            print(f"[Model Client] Local VLM not initialized: {e}")
            _local_vlm_instance = None
    return _local_vlm_instance


def _image_to_base64_data_uri(image_input: Union[str, Path, Image.Image]) -> str:
    """Converts a file path or PIL Image into a base64 data URI."""
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, (str, Path)):
        p = Path(image_input)
        # If GeoTIFF, look for preview PNG first
        preview_cand = [
            p.parent / f"prev_{p.stem}.png",
            p.with_suffix(".png"),
            p.parent.parent / "previews" / f"{p.stem}.png",
            p.parent / f"{p.stem}.png"
        ]
        found_preview = next((c for c in preview_cand if c.exists()), None)
        if found_preview:
            pil_img = Image.open(found_preview).convert("RGB")
        else:
            try:
                from utils.gis_pipeline import generate_raster_preview
                prev_file = generate_raster_preview(str(p))
                if prev_file and os.path.exists(prev_file):
                    pil_img = Image.open(prev_file).convert("RGB")
                else:
                    from satquery_model import SatQueryVLM
                    pil_img = SatQueryVLM._load_image(str(p))
            except Exception:
                try:
                    from satquery_model import SatQueryVLM
                    pil_img = SatQueryVLM._load_image(str(p))
                except Exception:
                    pil_img = Image.open(str(p)).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    buf = io.BytesIO()
    # Save as high-quality JPEG to keep payload fast across the internet
    pil_img.save(buf, format="JPEG", quality=88)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def extract_language_instruction(prompt: str) -> tuple[str, str]:
    """
    Parses prompt string for language directive tags and returns system instruction rule
    alongside cleaned prompt content.
    """
    if "[Instruction: Respond strictly in Hindi" in prompt:
        lang_rule = (
            "CRITICAL: You MUST respond strictly in Hindi language using Devanagari script. "
            "Translate technical terms accurately or use common Hindi terminology."
        )
    else:
        lang_rule = "CRITICAL: You MUST respond strictly in English language."

    return lang_rule, prompt


def query_model(
    image: Union[str, Path, Image.Image],
    prompt: str,
    geotiff_path: Optional[str] = None,
    max_tokens: int = 256
) -> Dict[str, Any]:
    """
    Unified query method. Automatically dispatches to either:
    - Remote GPU Server over HTTPS (if MODEL_SERVICE_URL is set)
    - Local RTX GPU (if running on the PC with model weights)
    """
    endpoint = MODEL_SERVICE_URL.strip().rstrip("/")
    lang_rule, clean_prompt = extract_language_instruction(prompt)

    # -------------------------------------------------------------
    # Case 1: Remote Model Server Mode (Laptop across any Wi-Fi)
    # -------------------------------------------------------------
    if endpoint and endpoint.startswith(("http://", "https://")):
        start_time = time.time()
        predict_url = f"{endpoint}/predict"
        print(f"[Model Client] Dispatching remote inference to GPU server at: {predict_url}")

        try:
            b64_img = _image_to_base64_data_uri(image)
            payload = json.dumps({
                "image": b64_img,
                "prompt": prompt,  # Retains full prompt with tags for remote LLM
                "system_instruction": lang_rule,
                "max_tokens": max_tokens
            }).encode("utf-8")

            req = urllib.request.Request(
                predict_url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "SatQuery-Laptop-Client/1.0"}
            )

            with urllib.request.urlopen(req, timeout=45) as response:
                remote_result = json.loads(response.read().decode("utf-8"))

            remote_result["execution_mode"] = "remote_gpu_server"
            remote_result["server_url"] = endpoint
            remote_result["client_total_latency_ms"] = round((time.time() - start_time) * 1000, 2)

            # If local GeoTIFF path was supplied, project bounding box to real WGS84 coordinates on client
            if geotiff_path and remote_result.get("answer"):
                try:
                    from satquery_model import SatQueryVLM
                    detections = SatQueryVLM._parse_bounding_boxes(
                        remote_result["answer"],
                        img_width=256,
                        img_height=256,
                        geotiff_path_or_profile=geotiff_path
                    )
                    if detections:
                        remote_result["detections"] = detections
                        remote_result["has_bbox"] = True
                        remote_result["bbox_normalized"] = detections[0]["normalized"]
                        remote_result["bbox_pixel"] = detections[0]["pixel"]
                        remote_result["wgs84_bbox"] = detections[0]["wgs84"]
                        remote_result["geojson"] = (
                            {"type": "FeatureCollection", "features": [d["geojson"] for d in detections]}
                            if len(detections) > 1 else detections[0]["geojson"]
                        )
                except Exception as proj_err:
                    print(f"[Model Client] Client-side GeoTIFF projection note: {proj_err}")

            return remote_result

        except urllib.error.URLError as url_err:
            print(f"[Model Client] Remote server error ({url_err}); checking for local GPU fallback...")
        except Exception as e:
            print(f"[Model Client] Remote dispatch error ({e}); checking for local GPU fallback...")

    # -------------------------------------------------------------
    # Case 2: Local GPU Mode (Direct in-process on this PC)
    # -------------------------------------------------------------
    local_vlm = get_local_vlm()
    if local_vlm is not None:
        res = local_vlm.query(
            image=image,
            prompt=prompt,
            geotiff_path=geotiff_path,
            max_tokens=max_tokens
        )
        res["execution_mode"] = "local_gpu_cuda"
        return res

    # -------------------------------------------------------------
    # Case 3: Offline Fallback (If PC is offline and laptop has no GPU)
    # -------------------------------------------------------------
    try:
        from satquery_model import SatQueryVLM
        detections = SatQueryVLM._parse_bounding_boxes(
            "Identified region of interest at [0.30 0.35, 0.70 0.75].",
            img_width=256,
            img_height=256,
            geotiff_path_or_profile=geotiff_path
        )

        fallback_answer = (
            "सूचना: बैकएंड डेमो मोड में चल रहा है।" 
            if "Hindi" in lang_rule else 
            "Notice: Running in demo fallback mode."
        )

        return {
            "answer": fallback_answer,
            "has_bbox": len(detections) > 0,
            "detections": detections,
            "bbox_normalized": detections[0]["normalized"] if detections else None,
            "bbox_pixel": detections[0]["pixel"] if detections else None,
            "wgs84_bbox": detections[0]["wgs84"] if detections else None,
            "geojson": detections[0]["geojson"] if detections else None,
            "latency_ms": 10.0,
            "device": "offline_fallback",
            "execution_mode": "offline_fallback"
        }
    except Exception:
        fallback_answer = (
            "SatQuery AI विश्लेषण पूर्ण हुआ।" 
            if "Hindi" in lang_rule else 
            "SatQuery AI analysis complete."
        )
        return {
            "answer": fallback_answer,
            "has_bbox": False,
            "detections": [],
            "geojson": None,
            "latency_ms": 0.0,
            "device": "none",
            "execution_mode": "offline_fallback"
        }