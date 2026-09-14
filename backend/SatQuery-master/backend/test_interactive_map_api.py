"""
Automated Test for Interactive Map STAC/COG API Router.
Uses FastAPI TestClient to test /api/v1/interactive-map endpoints.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    print("Testing /api/v1/interactive-map/health...")
    resp = client.get("/api/v1/interactive-map/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    data = resp.json()
    print("Health response:", data)
    assert data.get("status") == "online"
    assert data.get("module_loaded") is True
    print(">>> Health Check PASSED!\n")

def test_oversized_rejection():
    print("Testing oversized bounding box rejection...")
    # 70-80 lon, 20-30 lat is > 1 million km²
    resp = client.post("/api/v1/interactive-map/fetch", json={
        "bbox": [70.0, 20.0, 80.0, 30.0],
        "sensor": "sentinel-2"
    })
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    print("Oversized rejection message:", resp.json())
    print(">>> Oversized BBox Rejection PASSED!\n")

def test_inverted_rejection():
    print("Testing inverted coordinates rejection...")
    resp = client.post("/api/v1/interactive-map/fetch", json={
        "bbox": [78.0, 28.0, 77.0, 29.0],
        "sensor": "sentinel-2"
    })
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    print("Inverted rejection message:", resp.json())
    print(">>> Inverted Coords Rejection PASSED!\n")

def test_fetch_endpoint():
    print("Testing /api/v1/interactive-map/fetch with real Sentinel-2 pass...")
    resp = client.post("/api/v1/interactive-map/fetch", json={
        "bbox": [77.10, 28.55, 77.20, 28.65],
        "sensor": "sentinel-2",
        "max_cloud_cover": 20
    })
    assert resp.status_code == 200, f"Fetch failed: {resp.text}"
    data = resp.json()
    print("Fetch response keys:", list(data.keys()))
    print("Dataset ID:        ", data.get("dataset_id"))
    print("Name:              ", data.get("name"))
    print("Sensor:            ", data.get("sensor"))
    print("WGS84 Bounds:      ", data.get("wgs84_bounds"))
    print("T1 Image URL:      ", data.get("t1_image_url"))
    print("Suggested Queries: ", data.get("suggested_queries"))

    assert data.get("status") == "success"
    assert data.get("name", "").startswith("raw_S2_")
    assert data.get("t1_image_url", "").startswith("/static/")

    # Verify static file serving
    static_url = data.get("t1_image_url")
    static_resp = client.get(static_url)
    assert static_resp.status_code == 200, f"Failed to serve static overlay at {static_url}: {static_resp.status_code}"
    print(f"Static overlay served successfully: {len(static_resp.content)} bytes")
    return data.get("name")

def test_chatbot_handoff(geotiff_filename: str):
    print(f"Testing Chatbot query on streamed GeoTIFF: {geotiff_filename}...")
    resp = client.post("/api/v1/satquery/json", json={
        "query": "What is the primary land use in this satellite scene?",
        "dataset_name": geotiff_filename
    })
    assert resp.status_code == 200, f"Chatbot query failed: {resp.text}"
    result = resp.json()
    print("Chatbot Answer:", result.get("text_answer") or result.get("response") or "Analysis complete")
    print("Execution Trace:", len(result.get("execution_trace", {}).get("steps", [])))
    print(">>> Chatbot Query on Streamed GeoTIFF PASSED!\n")

if __name__ == "__main__":
    test_health()
    test_oversized_rejection()
    test_inverted_rejection()
    fetched_name = test_fetch_endpoint()
    if fetched_name:
        test_chatbot_handoff(fetched_name)
    print("ALL FASTAPI INTERACTIVE MAP ROUTER TESTS PASSED!")
