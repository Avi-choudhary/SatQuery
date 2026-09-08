import os
import json
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import METADATA_DIR, PREVIEWS_DIR, PATCHES_DIR
from src.dataloader import PatchIndex

app = FastAPI(
    title="SatQueryAI API",
    description="Geospatial satellite tile indexing, preview serving, and LLM-powered spatial querying.",
    version="1.0.0"
)

# Enable CORS for frontend web integration (Leaflet, Mapbox, React, Vue, Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global patch catalog index
patch_index = PatchIndex()


class ChatRequest(BaseModel):
    prompt: str
    region: Optional[str] = None
    date: Optional[str] = None
    sensor: Optional[str] = None


@app.get("/")
def root():
    """Service status and summary of indexed satellite data."""
    patch_index.reload()
    total = len(patch_index.records)
    regions = sorted(list({r.get("region") for r in patch_index.records if r.get("region")}))
    dates = sorted(list({r.get("date") for r in patch_index.records if r.get("date")}))
    sensors = sorted(list({r.get("sensor") for r in patch_index.records if r.get("sensor")}))

    return {
        "service": "SatQueryAI",
        "status": "online",
        "indexed_patches_count": total,
        "available_regions": regions,
        "available_dates": dates,
        "available_sensors": sensors,
        "endpoints": {
            "patches": "/api/patches",
            "preview": "/api/patches/{patch_name}/preview",
            "chat": "/api/chat"
        }
    }


@app.get("/api/patches")
def get_patches(
    region: Optional[str] = None,
    sensor: Optional[str] = None,
    date: Optional[str] = None,
    min_lon: Optional[float] = None,
    min_lat: Optional[float] = None,
    max_lon: Optional[float] = None,
    max_lat: Optional[float] = None,
    as_geojson: bool = True
):
    """
    Retrieve satellite patch metadata with optional spatial bounding box or attribute filters.
    Returns GeoJSON FeatureCollection by default for direct map rendering.
    """
    patch_index.reload()

    if all(coord is not None for coord in [min_lon, min_lat, max_lon, max_lat]):
        records = patch_index.query_bbox(
            min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat,
            region=region, sensor=sensor, date=date
        )
    else:
        records = patch_index.query(region=region, sensor=sensor, date=date)

    if as_geojson:
        return patch_index.to_geojson(records)
    return records


@app.get("/api/patches/{patch_name}")
def get_patch_metadata(patch_name: str):
    """Get metadata for a specific patch by name (e.g. delhi_20260112_S2_p0000)."""
    patch_index.reload()
    for r in patch_index.records:
        if r.get("name") == patch_name or str(r.get("patch_id")) == patch_name:
            return r
    raise HTTPException(status_code=404, detail=f"Patch '{patch_name}' not found.")


@app.get("/api/patches/{patch_name}/preview")
def get_patch_preview(patch_name: str):
    """Serve the 8-bit web-friendly PNG preview for browser rendering or map overlays."""
    # Strip extension if passed
    clean_name = patch_name.replace(".png", "").replace(".tif", "").replace(".json", "")
    png_path = PREVIEWS_DIR / f"{clean_name}.png"

    if not png_path.exists():
        # Check if saved directly in PATCHES_DIR
        alt_path = PATCHES_DIR / f"{clean_name}.png"
        if alt_path.exists():
            png_path = alt_path
        else:
            raise HTTPException(status_code=404, detail=f"Preview image for '{clean_name}' not found.")

    return FileResponse(png_path, media_type="image/png")


@app.post("/api/chat")
def chat_with_satellite_data(req: ChatRequest):
    """
    Natural language query interface for SatQueryAI.
    Combines LLM semantic interpretation with exact geospatial indexing.
    """
    patch_index.reload()
    prompt_lower = req.prompt.lower()

    # Detect sensor preference
    detected_sensor = req.sensor
    if not detected_sensor:
        if "sar" in prompt_lower or "radar" in prompt_lower or "sentinel-1" in prompt_lower or "s1" in prompt_lower:
            detected_sensor = "S1"
        elif "optical" in prompt_lower or "rgb" in prompt_lower or "sentinel-2" in prompt_lower or "s2" in prompt_lower:
            detected_sensor = "S2"

    # Query matching candidate records
    matched_records = patch_index.query(
        region=req.region,
        sensor=detected_sensor,
        date=req.date
    )

    # If prompt mentions specific landmarks or keywords, score/filter
    if "delhi" in prompt_lower and not req.region:
        matched_records = [r for r in matched_records if r.get("region") == "delhi"]

    # Limit to top relevant patches for display
    display_records = matched_records[:12] if matched_records else []
    geojson_result = patch_index.to_geojson(display_records)

    # Call Gemini LLM if API key is provided
    gemini_key = os.getenv("GEMINI_API_KEY")
    llm_explanation = None

    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            context_summary = {
                "total_matched": len(matched_records),
                "sample_patches": [
                    {
                        "name": r.get("name"),
                        "sensor": r.get("sensor"),
                        "date": r.get("date"),
                        "center": r.get("center"),
                        "wgs84_bounds": r.get("bounding_box")
                    }
                    for r in display_records[:5]
                ]
            }

            system_prompt = (
                "You are SatQueryAI, an expert satellite data analysis agent. "
                "Analyze the user's satellite image request and explain which patches "
                "were identified and what satellite characteristics (optical reflectance or SAR backscatter) "
                "they represent."
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {"role": "user", "parts": [
                        {"text": f"User query: {req.prompt}\n\nMatched Geospatial Context: {json.dumps(context_summary, indent=2)}"}
                    ]}
                ]
            )
            if response and response.text:
                llm_explanation = response.text
        except Exception as e:
            llm_explanation = f"LLM reasoning unavailable ({str(e)}). Fallback spatial match applied."

    if not llm_explanation:
        sensor_desc = "Sentinel-2 Optical (RGB + NIR)" if detected_sensor == "S2" else "Sentinel-1 SAR (Radar)" if detected_sensor == "S1" else "Optical & SAR multi-modal"
        llm_explanation = (
            f"Found {len(matched_records)} matching {sensor_desc} patches for your query. "
            f"The map has been updated with exact bounding coordinates and preview overlays."
        )

    return {
        "query": req.prompt,
        "response": llm_explanation,
        "matched_count": len(matched_records),
        "patches": [
            {
                "name": r.get("name"),
                "patch_id": r.get("patch_id"),
                "region": r.get("region"),
                "date": r.get("date"),
                "sensor": r.get("sensor"),
                "center": r.get("center"),
                "bounding_box": r.get("bounding_box"),
                "png_url": r.get("png_url")
            }
            for r in display_records
        ],
        "geojson": geojson_result
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
