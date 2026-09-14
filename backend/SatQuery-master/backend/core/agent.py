import os
import sys
import re
from pathlib import Path
from typing import List

from core.tracer import Tracer
from schemas.responses import SatQueryResponse
from services import vqa_service, ground_service, change_service
from utils import gis_pipeline
from core.query_planner import parse_query_plan


CHANGE_TERMS = (
    "change",
    "changes",
    "changed",
    "changing",
    "temporal",
    "bi-temporal",
    "bitemporal",
    "multitemporal",
    "multi-temporal",
    "between dates",
    "before and after",
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
    "shrinkage",
    "reduced",
    "reduction",
    "growth",
    "new building",
    "new buildings",
    "new road",
    "new roads",
    "urbanisation",
    "urbanization",
    "deforestation",
    "differ",
    "difference",
    "differences",
    "compare",
    "comparison",
    "बदलाव",
    "परिवर्तन",
    "अंतर",
    "तुलना"
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
    "ढूंढो",
    "खोजो",
    "चिह्नित"
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
    "निर्देशांक",
    "अक्षांश",
    "देशांतर",
    "स्थान"
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
    if file_count >= 2:
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
    """
    Central Agentic Controller execution flow:
    Stage 1: Input reception & file validation.
    Stage 2: GIS Pre-processing (co-registration / CRS alignment).
    Stage 3: Natural language intent parsing with language directive retention.
    Stage 4: Specialist AI tool dispatch (VQA, Grounding, Change Detection, or Metadata).
    Stage 5: Aggregation and auditable trace compilation.
    """
    tracer = Tracer()
    file_count = len(file_paths)
    file_names = ", ".join(os.path.basename(f) for f in file_paths) if file_paths else "none"

    # Detect language directive prepended by frontend toggle
    is_hindi = "[Instruction: Respond strictly in Hindi" in query or bool(re.search(r"[\u0900-\u097F]", query))
    clean_display_query = re.sub(r"\[Instruction:[\s\S]*?\]\n?", "", query).strip()

    tracer.append_log(f"Received query: '{clean_display_query}'")
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

    # Co-register pairs if multi-file change detection
    processed_paths = file_paths
    if task == "CHANGE_DETECTION" and file_count >= 2:
        try:
            processed_paths = gis_pipeline.align_geotiffs(file_paths)
        except Exception as e:
            tracer.append_log(f"Co-registration skipped: {e}")

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

                if is_hindi:
                    text_answer = (
                        f"भौगोलिक निर्देशांक और स्थान सीमा (Spatial Extent):\n\n"
                        f"• केंद्र निर्देशांक (Center): {c_lat:.4f}° N, {c_lon:.4f}° E\n"
                        f"• देशांतर सीमा (Longitude): {min_lon:.4f}° E से {max_lon:.4f}° E\n"
                        f"• अक्षांश सीमा (Latitude): {min_lat:.4f}° N से {max_lat:.4f}° N\n"
                        f"• स्थानिक संदर्भ (CRS): {crs_str}\n"
                        f"• रेजोल्यूशन (GSD): {res_str} प्रति पिक्सेल\n"
                        f"• सेंसर प्रकार: {sensor_str}\n\n"
                        f"इस उपग्रह दृश्य की सीमा रेखा मानचित्र (MapLibre viewport) पर प्रक्षेपित कर दी गई है।"
                    )
                else:
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
                if is_hindi:
                    text_answer = (
                        f"सूचना: फ़ाइल '{fname}' एक सामान्य पिक्सेल छवि है जिसमें भौगोलिक हेडर टैग (geotransform / CRS) शामिल नहीं हैं।\n\n"
                        f"सटीक अक्षांश और देशांतर देखने के लिए, कृपया एक जियोरेफ़रेंस्ड GeoTIFF (.tif) फ़ाइल अपलोड करें।"
                    )
                else:
                    text_answer = (
                        f"Notice: The file '{fname}' is a standard pixel image without embedded geospatial header tags (geotransform / CRS).\n\n"
                        f"To inspect precise latitude and longitude, please upload a georeferenced GeoTIFF (.tif) or include companion metadata."
                    )
                tracer.append_log("step 3: file lacks embedded geospatial header tags")

        elif task == "CHANGE_DETECTION":
            tracer.append_log("step 2: selecting Change Detective specialist")
            text_answer, visual_evidence = await change_service.run_inference(
                query,
                processed_paths,
                tracer,
            )

        elif task == "GROUNDING":
            tracer.append_log("step 2: selecting grounding specialist")
            text_answer, visual_evidence = await ground_service.run_inference(
                query,
                processed_paths,
                tracer,
            )

        else:
            tracer.append_log("step 2: selecting VQA specialist")
            text_answer, visual_evidence = await vqa_service.run_inference(
                query,
                processed_paths,
                tracer,
            )

    except Exception as exc:
        tracer.append_log(f"Agent execution error: {type(exc).__name__}: {exc}")
        text_answer = (
            f"विश्लेषण पूरा नहीं हो सका: {exc}" if is_hindi else
            f"SatQuery could not complete the requested analysis. Reason: {exc}"
        )
        visual_evidence = []

    tracer.append_log("step 3: assembling structured SatQuery response")

    trace_dict = tracer.get_trace()
    return SatQueryResponse(
        text_answer=text_answer,
        visual_evidence=visual_evidence,
        execution_trace=trace_dict,
        trace_log=trace_dict.get("steps", []),
    )
