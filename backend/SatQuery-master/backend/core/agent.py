import os
import re
from typing import List

from core.tracer import Tracer
from schemas.responses import SatQueryResponse
from services import vqa_service, ground_service, change_service
from utils import gis_pipeline
from core.query_planner import parse_query_plan


CHANGE_TERMS = (
    "change",
    "changed",
    "changes",
    "between",
    "before",
    "after",
    "temporal",
    "bi-temporal",
    "bitemporal",
    "multitemporal",
    "multi-temporal",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "gain",
    "gained",
    "loss",
    "lost",
    "expanded",
    "expansion",
    "shrunk",
    "reduced",
    "reduction",
    "growth",
    "new building",
    "built-up",
    "built up",
    "urbanisation",
    "urbanization",
    "urban",
    "water body",
    "water bodies",
    "water",
    "lake",
    "river",
    "reservoir",
    "vegetation",
    "vegetated",
    "greenery",
    "forest",
    "agriculture",
    "agricultural",
    "farming",
    "crop",
    "crops",
    "construction",
    "flooding",
    "flood",
    "inundation",
    "infrastructure",
    "road",
    "roads",
    "brightening",
    "darkening",
    "ndvi",
    "ndwi",
    "ndbi",
    "why",
    "cause",
    "differ",
    "difference",
    "differences",
    "compare",
    "comparison",
)

GROUNDING_TERMS = (
    "where",
    "location",
    "locate",
    "find",
    "highlight",
    "bbox",
    "bounding box",
    "outline",
    "polygon",
    "pinpoint",
)

COORD_TERMS = (
    "longitude",
    "latitude",
    "lattitude",
    "lat and lon",
    "lat/lon",
    "coordinates of",
    "coordinate of",
    "where is this image",
    "where is this scene",
    "geographic location",
    "bounding box of this",
    "spatial extent",
    "what are the coordinates",
    "gps coordinates",
    "crs",
)


def _contains_term(text: str, term: str) -> bool:
    return bool(
        re.search(
            rf"\b{re.escape(term)}\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_change_query(query: str, file_count: int) -> bool:
    if file_count == 2:
        return True
    text = (query or "").lower()
    return any(_contains_term(text, term) for term in CHANGE_TERMS)


def _looks_like_coord_query(query: str) -> bool:
    text = (query or "").lower()
    return any(k in text for k in COORD_TERMS)


def _looks_like_grounding_query(query: str) -> bool:
    text = (query or "").lower()
    return any(_contains_term(text, term) for term in GROUNDING_TERMS)


async def process_query(query: str, file_paths: List[str]) -> SatQueryResponse:
    tracer = Tracer()
    file_count = len(file_paths)
    file_names = ", ".join(os.path.basename(f) for f in file_paths) if file_paths else "none"

    tracer.append_log(f"Received query: '{query}'")
    tracer.append_log(f"Received {file_count} input file(s) [{file_names}]")

    # Step 1: Query Planner
    plan = parse_query_plan(query)
    tracer.append_log(
        f"step 1: query plan intent={plan.intent}, phenomenon={plan.phenomenon}, operation={plan.operation}, target_concept='{plan.target_concept}'"
    )

    # Step 2: Intent Classification & Routing
    if _looks_like_change_query(query, file_count):
        task = "CHANGE_DETECTION"
    elif _looks_like_coord_query(query) and file_count <= 1:
        task = "GEOSPATIAL_METADATA"
    elif _looks_like_grounding_query(query):
        task = "GROUNDING"
    else:
        task = "VQA"

    tracer.append_log(f"step 1b: classified task as {task}")

    visual_evidence = []
    text_answer = ""

    try:
        if task == "GEOSPATIAL_METADATA":
            tracer.append_log("step 2: routed to GIS Metadata & Coordinate Engine")
            target_file = file_paths[0] if file_paths else None
            info = gis_pipeline.get_geotiff_info(target_file) if target_file else {}

            if info.get("has_georeference") and info.get("wgs84_bounds"):
                min_lon, min_lat, max_lon, max_lat = info["wgs84_bounds"]
                c_lat = info.get("center_lat")
                c_lon = info.get("center_lon")
                crs_str = info.get("crs", "EPSG:4326 (WGS84)")
                sensor_str = info.get("sensor", "Sentinel-2 (Optical)")
                res_val = info.get("resolution")
                res_str = f"{res_val[0]}m" if res_val and isinstance(res_val, list) else "10.0m"

                text_answer = (
                    f"Geographic Coordinates & Spatial Extent:\n\n"
                    f"• Center Coordinates: {c_lat:.4f}° N, {c_lon:.4f}° E\n"
                    f"• Longitude Range: {min_lon:.4f}° E to {max_lon:.4f}° E\n"
                    f"• Latitude Range: {min_lat:.4f}° N to {max_lat:.4f}° N\n"
                    f"• Spatial Reference (CRS): {crs_str}\n"
                    f"• Resolution (GSD): {res_str} per pixel\n"
                    f"• Sensor: {sensor_str}\n\n"
                    f"The spatial boundary polygon for this scene has been projected onto the MapLibre viewport."
                )

                geojson_poly = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [min_lon, min_lat],
                            [max_lon, min_lat],
                            [max_lon, max_lat],
                            [min_lon, max_lat],
                            [min_lon, min_lat]
                        ]]
                    },
                    "properties": {
                        "label": f"{os.path.basename(target_file)} Boundary",
                        "crs": crs_str,
                        "center": [c_lon, c_lat]
                    }
                }
                visual_evidence = [geojson_poly]
                tracer.append_log(
                    f"step 3: extracted geographic bounds [{min_lat:.4f}°N, {min_lon:.4f}°E] from geospatial header"
                )
            else:
                fname = os.path.basename(target_file) if target_file else "uploaded file"
                text_answer = (
                    f"Notice: The file '{fname}' is a standard pixel image without embedded geospatial header tags (geotransform / CRS).\n\n"
                    f"To inspect precise latitude and longitude, please upload a georeferenced GeoTIFF (.tif) or include companion metadata."
                )
                tracer.append_log("step 3: file lacks embedded geospatial header tags")

        elif task == "CHANGE_DETECTION":
            tracer.append_log("step 2: selecting Change Detective specialist")
            text_answer, visual_evidence = await change_service.run_inference(
                query,
                file_paths,
                tracer,
            )

        elif task == "GROUNDING":
            tracer.append_log("step 2: selecting grounding specialist")
            text_answer, visual_evidence = await ground_service.run_inference(
                query,
                file_paths,
                tracer,
            )

        else:
            tracer.append_log("step 2: selecting VQA specialist")
            text_answer, visual_evidence = await vqa_service.run_inference(
                query,
                file_paths,
                tracer,
            )

    except Exception as exc:
        tracer.append_log(f"Agent execution error: {type(exc).__name__}: {exc}")
        text_answer = f"SatQuery could not complete the requested analysis. Reason: {exc}"
        visual_evidence = []

    tracer.append_log("step 3: assembling structured SatQuery response")

    trace_dict = tracer.get_trace()
    return SatQueryResponse(
        text_answer=text_answer,
        visual_evidence=visual_evidence,
        execution_trace=trace_dict,
        trace_log=trace_dict.get("steps", []),
    )
