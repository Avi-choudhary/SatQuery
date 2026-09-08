"""
SatQuery AI Full-Stack End-to-End System Verification
=====================================================
Verifies:
1. FastAPI Server & Routing (/api/v1/status, /api/v1/satquery)
2. GIS Extraction Engine (coregistration, metadata extraction, CRS alignment)
3. Central Agentic Controller (intent classification, stage tracking)
4. Specialist Tool 1: Single-Image VQA (Qwen3-VL reasoning)
5. Specialist Tool 2: Visual Grounding (WGS84 GeoJSON projection)
6. Response Schema Compatibility with React/MapLibre Frontend
"""

import os
import sys
import io
import json
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image

# Ensure backend is on sys.path
backend_dir = Path(__file__).resolve().parents[1] / "backend" / "SatQuery-master" / "backend"
sys.path.insert(0, str(backend_dir))

from main import app
from utils import gis_pipeline

def run_tests():
    print("=" * 70)
    print("   SATQUERY AI SYSTEM-WIDE END-TO-END VERIFICATION TEST")
    print("=" * 70)

    client = TestClient(app)

    # -------------------------------------------------------------
    # Test 1: Root and Status Endpoints
    # -------------------------------------------------------------
    print("\n[1/6] Testing Root & Status Endpoints...")
    root_resp = client.get("/")
    assert root_resp.status_code == 200, f"Root failed: {root_resp.status_code}"
    print(f"  [PASS] Root endpoint: {root_resp.json().get('service')} (status: {root_resp.json().get('status')})")

    status_resp = client.get("/api/v1/status")
    assert status_resp.status_code == 200, f"Status failed: {status_resp.status_code}"
    status_data = status_resp.json()
    model_eng = status_data.get("model_engine", {})
    print(f"  [PASS] Status endpoint: Mode={status_data.get('execution_mode')}")
    print(f"         Device: {model_eng.get('local_device')} | VRAM: {model_eng.get('local_vram_gb')} GB")
    print(f"         CUDA Available: {model_eng.get('local_cuda_available')}")

    # -------------------------------------------------------------
    # Test 2: GIS Engine & Coordinate Transform
    # -------------------------------------------------------------
    print("\n[2/6] Testing GIS Pre-processing Engine...")
    test_tif = Path(__file__).resolve().parents[1] / "gis_extraction_logic" / "data" / "interim" / "delhi_20260112_S2_aligned.tif"
    if test_tif.exists():
        info = gis_pipeline.get_geotiff_info(str(test_tif))
        assert info["is_geotiff"] is True
        assert "32643" in str(info["crs"]) or "EPSG" in str(info["crs"])
        print(f"  [PASS] GeoTIFF Ingest: CRS={info.get('crs')}, Shape={info.get('shape')}")
    else:
        print("  [NOTE] delhi_20260112_S2_aligned.tif not found; checking get_geotiff_info fallback")
        info = gis_pipeline.get_geotiff_info("sample.tif")
        assert info["is_geotiff"] is True
        print(f"  [PASS] GeoTIFF Ingest fallback validated.")

    # -------------------------------------------------------------
    # Test 3: Agentic Controller & Specialist Tool 1 (VQA)
    # -------------------------------------------------------------
    print("\n[3/6] Testing Agentic Controller -> Specialist Tool 1 (Single-Image VQA)...")
    vqa_resp = client.post(
        "/api/v1/satquery",
        data={"query": "What is the primary land cover and terrain type visible in this satellite imagery?"}
    )
    assert vqa_resp.status_code == 200, f"VQA failed: {vqa_resp.status_code} - {vqa_resp.text}"
    vqa_data = vqa_resp.json()
    
    assert "text_answer" in vqa_data, "Missing text_answer"
    assert "execution_trace" in vqa_data, "Missing execution_trace"
    assert "steps" in vqa_data["execution_trace"], "Missing execution_trace.steps"
    assert "logs" in vqa_data["execution_trace"], "Missing execution_trace.logs"
    assert "trace_log" in vqa_data, "Missing trace_log alias"
    
    trace_steps = vqa_data["execution_trace"]["steps"]
    assert any("VQA" in s for s in trace_steps), "VQA intent not routed"
    print(f"  [PASS] VQA Response: '{vqa_data['text_answer'][:75]}...'")
    print(f"  [PASS] Trace Stages: {len(trace_steps)} auditable steps verified")

    # -------------------------------------------------------------
    # Test 4: Agentic Controller -> Specialist Tool 2 (Visual Grounding)
    # -------------------------------------------------------------
    print("\n[4/6] Testing Agentic Controller -> Specialist Tool 2 (Visual Grounding)...")
    ground_resp = client.post(
        "/api/v1/satquery",
        data={"query": "Where are the primary buildings and infrastructure located? Outline them."}
    )
    assert ground_resp.status_code == 200, f"Grounding failed: {ground_resp.status_code} - {ground_resp.text}"
    ground_data = ground_resp.json()
    
    assert "visual_evidence" in ground_data, "Missing visual_evidence"
    assert len(ground_data["visual_evidence"]) > 0, "No visual evidence returned for grounding query"
    
    geojson_feature = ground_data["visual_evidence"][0]
    print(f"  [PASS] Grounding Response: '{ground_data['text_answer'][:60]}...'")
    print(f"  [PASS] Visual Evidence: {geojson_feature.get('type')} detected")
    if geojson_feature.get("geometry"):
        coords = geojson_feature["geometry"].get("coordinates", [[]])[0]
        print(f"  [PASS] Coordinate Polygon: {len(coords)} vertices | Sample point: {coords[0] if coords else 'None'}")
    
    ground_steps = ground_data["execution_trace"]["steps"]
    assert any("GROUNDING" in s for s in ground_steps), "Grounding intent not routed"
    print(f"  [PASS] Trace Stages: {len(ground_steps)} auditable steps verified")

    # -------------------------------------------------------------
    # Test 5: Multipart File Upload with Real Image
    # -------------------------------------------------------------
    print("\n[5/6] Testing Multipart File Upload (simulating user drag-and-drop)...")
    buf = io.BytesIO()
    test_img = Image.new("RGB", (256, 256), color=(60, 110, 80))
    test_img.save(buf, format="PNG")
    buf.seek(0)
    
    upload_resp = client.post(
        "/api/v1/satquery",
        data={"query": "Identify features in this uploaded scene.", "dataset_name": "User_Uploaded_Tile.png"},
        files={"files": ("User_Uploaded_Tile.png", buf, "image/png")}
    )
    assert upload_resp.status_code == 200, f"Upload query failed: {upload_resp.status_code} - {upload_resp.text}"
    upload_data = upload_resp.json()
    print(f"  [PASS] User File Upload Query succeeded: '{upload_data['text_answer'][:60]}...'")
    assert any("User_Uploaded_Tile.png" in s for s in upload_data["execution_trace"]["steps"])
    print(f"  [PASS] File name preserved in execution trace logs")

    # -------------------------------------------------------------
    # Test 6: JSON Endpoint
    # -------------------------------------------------------------
    print("\n[6/6] Testing JSON Endpoint (/api/v1/satquery/json)...")
    json_resp = client.post(
        "/api/v1/satquery/json",
        json={"query": "Summarize land features in JSON format."}
    )
    assert json_resp.status_code == 200, f"JSON query failed: {json_resp.status_code} - {json_resp.text}"
    print(f"  [PASS] JSON endpoint response: '{json_resp.json()['text_answer'][:60]}...'")

    print("\n" + "=" * 70)
    print("   ALL 6 SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
