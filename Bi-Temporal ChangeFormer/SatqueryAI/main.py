from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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