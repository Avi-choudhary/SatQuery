from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# Mount the static directory to serve images like heatmaps
app.mount("/static/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="static_outputs")


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