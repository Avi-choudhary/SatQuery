"""
SatQuery Model API Microservice
================================
Lightweight FastAPI server exposing the fine-tuned SatQuery Qwen3-VL model.

Run standalone via:
    python serve_model.py --port 8008

Endpoints:
- GET  /health          : Status and GPU VRAM utilization
- POST /predict         : Main JSON / Multipart image prediction
- POST /api/generate    : Ollama-compatible endpoint
- POST /api/chat        : Ollama-compatible chat endpoint
"""

import os
import argparse
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch

from satquery_model import SatQueryVLM

app = FastAPI(
    title="SatQuery Vision-Language Model Service",
    description="Dedicated Earth Observation VLM inference microservice for SatQuery AI (SIH 2026 / ISRO)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
vlm: Optional[SatQueryVLM] = None


class PredictRequest(BaseModel):
    image: str  # File path or base64 string
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.2


class OllamaGenerateRequest(BaseModel):
    model: Optional[str] = "satquery-qwen:2b"
    prompt: str
    images: Optional[List[str]] = None  # Base64 strings
    stream: Optional[bool] = False


@app.on_event("startup")
def startup_event():
    global vlm
    print("\n[Server Startup] Loading SatQuery VLM into GPU memory...")
    vlm = SatQueryVLM(warmup=False)
    print("[Server Startup] Model ready to serve requests.\n")


@app.get("/health")
def health():
    cuda_avail = torch.cuda.is_available()
    vram_alloc = round(torch.cuda.memory_allocated(0) / 1024**3, 2) if cuda_avail else 0
    vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if cuda_avail else 0

    return {
        "status": "ready" if vlm is not None else "loading",
        "model": "Qwen3-VL-2B-SatQuery-LoRA",
        "device": str(vlm.device) if vlm else "unknown",
        "device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
        "vram_allocated_gb": vram_alloc,
        "vram_total_gb": vram_total,
        "vram_free_gb": round(vram_total - vram_alloc, 2)
    }


@app.post("/predict")
async def predict(
    request: Optional[PredictRequest] = None,
    file: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form(None),
    max_tokens: int = Form(256)
):
    """
    Accepts either JSON {image, prompt} or multipart form-data (file + prompt).
    """
    global vlm
    if vlm is None:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    # Case 1: Multipart upload
    if file is not None:
        img_bytes = await file.read()
        q = prompt or "Analyze this Sentinel-2 satellite image and describe the land cover."
        return vlm.query(img_bytes, q, max_tokens=max_tokens)

    # Case 2: JSON payload
    if request is not None:
        return vlm.query(
            image=request.image,
            prompt=request.prompt,
            max_tokens=request.max_tokens or 256,
            temperature=request.temperature or 0.2
        )

    raise HTTPException(status_code=400, detail="Must provide either JSON payload or multipart form-data.")


# --- Ollama Compatibility Endpoints ---
@app.post("/api/generate")
async def ollama_generate(req: OllamaGenerateRequest):
    """
    Compatibility layer matching Ollama's POST /api/generate format.
    Allows existing Ollama-based backend code to connect seamlessly.
    """
    global vlm
    if vlm is None:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    if not req.images:
        raise HTTPException(status_code=400, detail="SatQuery VLM requires at least one image in 'images' list.")

    # Use first base64 image
    img_b64 = req.images[0]
    result = vlm.query(img_b64, req.prompt)

    return {
        "model": "satquery-qwen:2b",
        "created_at": "",
        "response": result["answer"],
        "done": True,
        "total_duration": int(result["latency_ms"] * 1e6),
        "satquery_meta": result
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start SatQuery Model Microservice")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()

    print(f"Starting SatQuery Model API Server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
