import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure current directory is on sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from api.routes import router as api_router
except ImportError:
    from backend.api.routes import router as api_router

from core.config import HOST, PORT, MODEL_SERVICE_URL, get_lan_ip

app = FastAPI(
    title="SatQuery AI Backend",
    description="Central Agentic Controller for Satellite Remote Sensing (SIH 2026 / ISRO PS 26167)",
    version="1.0.0"
)

# Configure CORS so any laptop on the LAN or local Vite dev server can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

app.include_router(api_router, prefix="/api/v1")

# Mount static directory for preview imagery & uploaded rasters
static_dir = os.path.join(current_dir, "temp_uploads")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount output directory for ChangeFormer generated masks, rasters, and GeoJSON files
candidate_output_dirs = [
    current_dir.parents[2] / "Bi-Temporal ChangeFormer" / "SatqueryAI" / "backend" / "outputs",
    current_dir.parents[1] / "Bi-Temporal ChangeFormer" / "SatqueryAI" / "backend" / "outputs",
    Path(r"c:\Games\SatQuery\Bi-Temporal ChangeFormer\SatqueryAI\backend\outputs"),
]
outputs_dir = candidate_output_dirs[0]
for cand in candidate_output_dirs:
    if cand.parent.exists():
        outputs_dir = cand
        break
os.makedirs(outputs_dir, exist_ok=True)
app.mount("/static/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")


@app.get("/")
def root():
    lan_ip = get_lan_ip()
    return {
        "service": "SatQuery AI Backend",
        "status": "online",
        "local_url": f"http://localhost:{PORT}/api/v1",
        "documentation": f"http://localhost:{PORT}/docs",
        "remote_gpu_url": MODEL_SERVICE_URL if MODEL_SERVICE_URL else "Running on local GPU"
    }


@app.get("/api/v1/status")
def system_status():
    """System status check reporting GPU, remote server connection, and active AI specialists."""
    lan_ip = get_lan_ip()
    gpu_available = False
    gpu_name = "None (CPU)"
    vram_gb = 0.0

    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    except Exception:
        pass

    remote_status = "not_configured"
    remote_vram = None
    if MODEL_SERVICE_URL:
        import urllib.request
        import json
        try:
            health_url = f"{MODEL_SERVICE_URL.rstrip('/')}/health"
            req = urllib.request.Request(health_url, headers={"User-Agent": "SatQueryStatus/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                remote_status = data.get("status", "online")
                remote_vram = f"{data.get('vram_allocated_gb', 0)} / {data.get('vram_total_gb', 0)} GB ({data.get('device_name', 'GPU')})"
        except Exception as e:
            remote_status = f"offline or unreachable ({type(e).__name__})"

    return {
        "service": "SatQuery AI",
        "status": "online",
        "execution_mode": "remote_gpu_api" if MODEL_SERVICE_URL else ("local_gpu_cuda" if gpu_available else "offline_fallback"),
        "model_engine": {
            "remote_service_url": MODEL_SERVICE_URL if MODEL_SERVICE_URL else None,
            "remote_host_status": remote_status if MODEL_SERVICE_URL else "N/A (running locally)",
            "remote_vram": remote_vram,
            "local_cuda_available": gpu_available,
            "local_device": gpu_name,
            "local_vram_gb": vram_gb
        },
        "network": {
            "lan_ip": lan_ip,
            "port": PORT,
            "api_endpoint": f"http://localhost:{PORT}/api/v1/satquery"
        },
        "specialists": {
            "tool_1_vqa": "ready (Qwen3-VL-2B-SatQuery)",
            "tool_2_grounding": "ready (WGS84 GeoJSON polygon projection)",
            "tool_3_change_detection": "ready (ChangeFormerV6 + Change Detective)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    lan_ip = get_lan_ip()
    print("\n" + "=" * 60)
    print("SatQuery AI Central Backend Starting...")
    print(f"Local Access:  http://localhost:{PORT}")
    print(f"LAN Access:    http://{lan_ip}:{PORT}  <-- Use this from other laptops")
    print(f"API Docs:      http://localhost:{PORT}/docs")
    print("=" * 60 + "\n")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
