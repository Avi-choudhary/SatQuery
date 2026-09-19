from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

from api.routes import router

app = FastAPI(
    title="SatQuery AI",
    version="1.0.0",
    description="Agentic satellite imagery analysis system.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

# Ensure outputs directory exists
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

@app.get("/static/outputs/{file_path:path}")
async def serve_static_outputs(file_path: str):
    file = OUTPUTS_DIR / file_path
    if not file.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file))


@app.get("/")
async def root():
    return {
        "service": "SatQuery AI",
        "status": "ok",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
    }