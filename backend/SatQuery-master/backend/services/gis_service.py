from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyproj import CRS, Transformer
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy.ndimage import find_objects, label


def pixel_area_m2(transform: Affine) -> float:
    """
    Calculate the area represented by one raster pixel.

    This assumes the analysis raster uses a projected CRS whose units
    are metres.
    """

    return abs(
        float(transform.a)
        * float(transform.e)
    )


def mask_area_m2(
    mask: np.ndarray,
    transform: Affine,
) -> float:
    """
    Calculate total area represented by a binary raster mask.
    """

    changed_pixels = int(
        np.count_nonzero(mask)
    )

    return (
        changed_pixels
        * pixel_area_m2(transform)
    )


def mask_area_hectares(
    mask: np.ndarray,
    transform: Affine,
) -> float:
    """
    Calculate mask area in hectares.
    """

    return (
        mask_area_m2(
            mask,
            transform,
        )
        / 10_000.0
    )


def raster_bounds(
    transform: Affine,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """
    Return raster bounds as:
        left, bottom, right, top
    """

    corners = [
        transform * (0, 0),
        transform * (width, 0),
        transform * (0, height),
        transform * (width, height),
    ]

    xs = [
        point[0]
        for point in corners
    ]

    ys = [
        point[1]
        for point in corners
    ]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )


def polygonize_mask(
    mask: np.ndarray,
    transform: Affine,
    crs: CRS,
) -> List[Dict[str, Any]]:
    """
    Convert a binary raster change mask into GeoJSON-compatible features.

    Geometry is initially produced in the projected analysis CRS.
    """

    features: List[
        Dict[str, Any]
    ] = []

    pixel_area = pixel_area_m2(
        transform
    )

    for geometry, value in shapes(
        mask.astype(np.uint8),
        mask=mask.astype(bool),
        transform=transform,
    ):

        if int(value) != 1:
            continue

        area = _geometry_area(
            geometry
        )

        # Raster-generated polygons should have an area close to an
        # integer number of pixels. The exact polygon area is used here.
        if area <= 0:
            continue

        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "area_m2": float(area),
                    "area_ha": float(
                        area / 10_000.0
                    ),
                    "pixel_area_m2": float(
                        pixel_area
                    ),
                },
            }
        )

    features.sort(
        key=lambda feature: feature[
            "properties"
        ]["area_m2"],
        reverse=True,
    )

    return features


def _ring_area(
    ring: List[List[float]],
) -> float:
    """
    Shoelace area for a linear ring.
    """

    if len(ring) < 3:
        return 0.0

    area = 0.0

    for i in range(
        len(ring) - 1
    ):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]

        area += (
            x1 * y2
            - x2 * y1
        )

    return abs(area) / 2.0


def _geometry_area(
    geometry: Dict[str, Any],
) -> float:
    """
    Calculate polygon/multipolygon area in the geometry's CRS units.
    """

    geometry_type = geometry.get(
        "type"
    )

    coordinates = geometry.get(
        "coordinates"
    )

    if not coordinates:
        return 0.0

    if geometry_type == "Polygon":

        outer_area = _ring_area(
            coordinates[0]
        )

        hole_area = sum(
            _ring_area(ring)
            for ring in coordinates[1:]
        )

        return max(
            0.0,
            outer_area - hole_area,
        )

    if geometry_type == "MultiPolygon":

        total = 0.0

        for polygon in coordinates:

            if not polygon:
                continue

            outer_area = _ring_area(
                polygon[0]
            )

            hole_area = sum(
                _ring_area(ring)
                for ring in polygon[1:]
            )

            total += max(
                0.0,
                outer_area - hole_area,
            )

        return total

    return 0.0


def _ring_perimeter(ring: List[List[float]]) -> float:
    if len(ring) < 2:
        return 0.0
    perimeter = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        perimeter += math.hypot(x2 - x1, y2 - y1)
    return perimeter


def _geometry_perimeter(geometry: Dict[str, Any]) -> float:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if not coordinates:
        return 0.0
    if geometry_type == "Polygon":
        return sum(_ring_perimeter(ring) for ring in coordinates)
    if geometry_type == "MultiPolygon":
        return sum(sum(_ring_perimeter(ring) for ring in poly) for poly in coordinates if poly)
    return 0.0


def _ring_centroid(ring: List[List[float]]) -> Tuple[float, float, float]:
    if len(ring) < 3:
        if not ring:
            return 0.0, 0.0, 0.0
        return ring[0][0], ring[0][1], 0.0
    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    area = area / 2.0
    if abs(area) < 1e-9:
        xs = [pt[0] for pt in ring[:-1]]
        ys = [pt[1] for pt in ring[:-1]]
        if not xs or not ys:
            return 0.0, 0.0, 0.0
        return sum(xs) / len(xs), sum(ys) / len(ys), 0.0
    cx = cx / (6.0 * area)
    cy = cy / (6.0 * area)
    return cx, cy, abs(area)


def _geometry_centroid(geometry: Dict[str, Any]) -> Tuple[float, float]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if not coordinates:
        return 0.0, 0.0
    if geometry_type == "Polygon":
        cx, cy, _ = _ring_centroid(coordinates[0])
        return cx, cy
    if geometry_type == "MultiPolygon":
        total_a = 0.0
        sum_cx = 0.0
        sum_cy = 0.0
        for poly in coordinates:
            if not poly:
                continue
            cx, cy, a = _ring_centroid(poly[0])
            sum_cx += cx * a
            sum_cy += cy * a
            total_a += a
        if total_a > 1e-9:
            return sum_cx / total_a, sum_cy / total_a
        if coordinates and coordinates[0] and coordinates[0][0]:
            return coordinates[0][0][0][0], coordinates[0][0][0][1]
    return 0.0, 0.0


def _ring_bbox(ring: List[List[float]]) -> Tuple[float, float, float, float]:
    if not ring:
        return 0.0, 0.0, 0.0, 0.0
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    return min(xs), min(ys), max(xs), max(ys)


def geometry_bbox(geometry: Dict[str, Any]) -> Tuple[float, float, float, float]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if not coordinates:
        return 0.0, 0.0, 0.0, 0.0
    if geometry_type == "Polygon":
        return _ring_bbox(coordinates[0])
    if geometry_type == "MultiPolygon":
        boxes = [_ring_bbox(poly[0]) for poly in coordinates if poly]
        if not boxes:
            return 0.0, 0.0, 0.0, 0.0
        return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)
    return 0.0, 0.0, 0.0, 0.0


def classify_region_change(
    mean_before: np.ndarray,
    mean_after: np.ndarray,
    first_bands: Dict[str, int],
    second_bands: Dict[str, int],
    modality: str = "optical",
) -> Tuple[str, str, str, Dict[str, float]]:
    spectral_metrics: Dict[str, float] = {}

    if modality == "sar":
        delta_val = float(mean_after[0] - mean_before[0]) if len(mean_after) > 0 and len(mean_before) > 0 else 0.0
        spectral_metrics["delta_sar_channel0"] = round(delta_val, 4)
        if delta_val > 1.5:
            return (
                "Radar Backscatter Increase",
                "backscatter_increase",
                "Observed prominent radar backscatter enhancement. Plausible causes: Natural (increased surface roughness, soil moisture surge) or Anthropogenic (new metallic or vertical structures, double-bounce building reflection).",
                spectral_metrics,
            )
        if delta_val < -1.5:
            return (
                "Radar Backscatter Decrease",
                "backscatter_decrease",
                "Observed prominent radar backscatter reduction. Plausible causes: Natural (standing water flooding specular reflection, smooth mud deposition) or Anthropogenic (structure demolition, land flattening).",
                spectral_metrics,
            )
        return (
            "Radar Amplitude Variation",
            "backscatter_variation",
            "Observed minor radar backscatter variation. Plausible causes: Natural environmental fluctuation or soil moisture transition.",
            spectral_metrics,
        )

    has_nir = (
        "nir" in first_bands
        and first_bands["nir"] is not None
        and "nir" in second_bands
        and second_bands["nir"] is not None
        and "red" in first_bands
        and first_bands["red"] is not None
        and "red" in second_bands
        and second_bands["red"] is not None
    )

    if has_nir:
        nir1 = float(mean_before[first_bands["nir"] - 1]) if first_bands["nir"] <= len(mean_before) else 0.0
        red1 = float(mean_before[first_bands["red"] - 1]) if first_bands["red"] <= len(mean_before) else 0.0
        nir2 = float(mean_after[second_bands["nir"] - 1]) if second_bands["nir"] <= len(mean_after) else 0.0
        red2 = float(mean_after[second_bands["red"] - 1]) if second_bands["red"] <= len(mean_after) else 0.0

        ndvi1 = (nir1 - red1) / max(nir1 + red1, 1e-6)
        ndvi2 = (nir2 - red2) / max(nir2 + red2, 1e-6)
        delta_ndvi = ndvi2 - ndvi1

        spectral_metrics["ndvi_before"] = round(ndvi1, 4)
        spectral_metrics["ndvi_after"] = round(ndvi2, 4)
        spectral_metrics["delta_ndvi"] = round(delta_ndvi, 4)

        if delta_ndvi <= -0.15:
            return (
                "Vegetation Loss",
                "vegetation_loss",
                "Observed significant reduction in vegetative greenness (delta NDVI < -0.15). Plausible causes: Natural (seasonal crop harvest, drought dieback, wildfire scar) or Anthropogenic (land clearing, construction excavation, tree felling). Multitemporal vegetation profile recommended.",
                spectral_metrics,
            )
        if delta_ndvi >= 0.15:
            return (
                "Vegetation Gain",
                "vegetation_gain",
                "Observed significant increase in vegetative vigor (delta NDVI > +0.15). Plausible causes: Natural (seasonal green-up, monsoon flush) or Anthropogenic (agricultural crop growth, plantation/reforestation).",
                spectral_metrics,
            )

    has_water_indices = (
        has_nir
        and "green" in first_bands
        and first_bands["green"] is not None
        and "green" in second_bands
        and second_bands["green"] is not None
    )

    if has_water_indices:
        g1 = float(mean_before[first_bands["green"] - 1]) if first_bands["green"] <= len(mean_before) else 0.0
        g2 = float(mean_after[second_bands["green"] - 1]) if second_bands["green"] <= len(mean_after) else 0.0
        nir1 = float(mean_before[first_bands["nir"] - 1]) if first_bands["nir"] <= len(mean_before) else 0.0
        nir2 = float(mean_after[second_bands["nir"] - 1]) if second_bands["nir"] <= len(mean_after) else 0.0

        ndwi1 = (g1 - nir1) / max(g1 + nir1, 1e-6)
        ndwi2 = (g2 - nir2) / max(g2 + nir2, 1e-6)
        delta_ndwi = ndwi2 - ndwi1

        spectral_metrics["ndwi_before"] = round(ndwi1, 4)
        spectral_metrics["ndwi_after"] = round(ndwi2, 4)
        spectral_metrics["delta_ndwi"] = round(delta_ndwi, 4)

        if delta_ndwi >= 0.15:
            return (
                "Water Surface Expansion",
                "water_expansion",
                "Observed significant increase in surface wetness index. Plausible causes: Natural (monsoon flooding, riverine overflow, reservoir filling) or Anthropogenic (irrigation inundation, canal recharge).",
                spectral_metrics,
            )
        if delta_ndwi <= -0.15:
            return (
                "Water Surface Recession",
                "water_recession",
                "Observed significant decrease in surface wetness index. Plausible causes: Natural (seasonal waterbody drawdown, evapotranspiration) or Anthropogenic (water drainage, reservoir discharge).",
                spectral_metrics,
            )

    delta_channels = [float(a - b) for a, b in zip(mean_after[:3], mean_before[:3])]
    delta_brightness = float(sum(delta_channels) / max(len(delta_channels), 1))
    spectral_metrics["delta_brightness"] = round(delta_brightness, 4)

    if delta_brightness >= 0.06:
        return (
            "Surface Brightening",
            "surface_brightening",
            "Observed noticeable surface brightening across visible bands. Plausible causes: Natural (soil drying, seasonal crop residue exposure, senescing vegetation) or Anthropogenic (ground clearing, bare soil excavation, new construction activity). High-resolution multispectral verification recommended.",
            spectral_metrics,
        )
    if delta_brightness <= -0.06:
        return (
            "Surface Darkening",
            "surface_darkening",
            "Observed noticeable surface darkening across visible bands. Plausible causes: Natural (soil wetting/inundation, cloud shadow artifact, fire burn scar) or Anthropogenic (asphalt/bitumen road surfacing, building foundation completion). Multi-temporal contextual check recommended.",
            spectral_metrics,
        )

    return (
        "Visible Spectral Modification",
        "spectral_modification",
        "Observed localized chromatic variation across visible bands. Plausible causes: Natural phenological fluctuation or minor surface disturbance.",
        spectral_metrics,
    )


def polygonize_and_analyze_regions(
    mask: np.ndarray,
    transform: Affine,
    crs: CRS,
    magnitude: np.ndarray,
    before_data: np.ndarray,
    after_data: np.ndarray,
    first_bands: Dict[str, int],
    second_bands: Dict[str, int],
    modality: str = "optical",
    changeformer_mask: Optional[np.ndarray] = None,
    changeformer_status: str = "",
    registration_shift_magnitude: float = 0.0,
    radiometric_method: str = "identity",
    scene_threshold: float = 0.15,
) -> List[Dict[str, Any]]:
    lbl, num_features = label(mask)
    if num_features == 0:
        return []

    slices = find_objects(lbl)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    features: List[Dict[str, Any]] = []

    for geometry, value in shapes(
        lbl.astype(np.int32),
        mask=mask,
        transform=transform,
    ):
        label_id = int(value)
        if label_id <= 0 or label_id > len(slices):
            continue

        sl = slices[label_id - 1]
        if sl is None:
            continue

        region_pixel_mask = lbl[sl] == label_id
        if not np.any(region_pixel_mask):
            continue

        area_m2 = _geometry_area(geometry)
        if area_m2 <= 0:
            continue

        area_ha = area_m2 / 10_000.0
        perimeter_m = _geometry_perimeter(geometry)
        compactness = min(1.0, max(0.0, 4.0 * math.pi * area_m2 / max(perimeter_m * perimeter_m, 1e-7)))

        cx_proj, cy_proj = _geometry_centroid(geometry)
        lon, lat = transformer.transform(cx_proj, cy_proj)

        bbox_proj = geometry_bbox(geometry)
        bbox_p1_lon, bbox_p1_lat = transformer.transform(bbox_proj[0], bbox_proj[1])
        bbox_p2_lon, bbox_p2_lat = transformer.transform(bbox_proj[2], bbox_proj[3])
        bbox_wgs84 = [
            round(float(min(bbox_p1_lon, bbox_p2_lon)), 6),
            round(float(min(bbox_p1_lat, bbox_p2_lat)), 6),
            round(float(max(bbox_p1_lon, bbox_p2_lon)), 6),
            round(float(max(bbox_p1_lat, bbox_p2_lat)), 6),
        ]

        reg_mag = magnitude[sl][region_pixel_mask]
        finite_reg_mag = reg_mag[np.isfinite(reg_mag)]
        mean_mag = float(np.mean(finite_reg_mag)) if finite_reg_mag.size > 0 else 0.0
        max_mag = float(np.max(finite_reg_mag)) if finite_reg_mag.size > 0 else 0.0

        b_crop = before_data[:, sl[0], sl[1]][:, region_pixel_mask]
        a_crop = after_data[:, sl[0], sl[1]][:, region_pixel_mask]

        mean_before = np.mean(b_crop, axis=1) if b_crop.size > 0 else np.zeros(before_data.shape[0])
        mean_after = np.mean(a_crop, axis=1) if a_crop.size > 0 else np.zeros(after_data.shape[0])

        change_name, change_type, hypothesis, metrics = classify_region_change(
            mean_before,
            mean_after,
            first_bands,
            second_bands,
            modality=modality,
        )

        area_rating = "STRONG" if area_ha >= 0.10 else ("MODERATE" if area_ha >= 0.04 else "WEAK")
        mag_ratio = mean_mag / max(scene_threshold, 1e-6)
        mag_rating = "STRONG" if mag_ratio >= 2.0 else ("MODERATE" if mag_ratio >= 1.25 else "WEAK")
        reg_rating = "STRONG" if registration_shift_magnitude < 1.0 else ("MODERATE" if registration_shift_magnitude < 2.5 else "WEAK")
        comp_rating = "STRONG" if compactness >= 0.15 else ("MODERATE" if compactness >= 0.05 else "WEAK")

        disagreement = False
        cf_rating = "NOT_EVALUATED"
        if changeformer_mask is not None:
            cf_crop = changeformer_mask[sl][region_pixel_mask]
            cf_fraction = float(np.count_nonzero(cf_crop)) / max(float(cf_crop.size), 1.0)
            if cf_fraction >= 0.30:
                cf_rating = "STRONG"
            elif cf_fraction < 0.10:
                cf_rating = "WEAK"
                disagreement = True
            else:
                cf_rating = "MODERATE"

        ratings = [area_rating, mag_rating, reg_rating, comp_rating]
        if cf_rating in ("STRONG", "MODERATE", "WEAK"):
            ratings.append(cf_rating)

        weak_count = ratings.count("WEAK")
        strong_count = ratings.count("STRONG")

        if weak_count >= 2 or registration_shift_magnitude >= 3.0:
            reliability_label = "WEAK"
        elif strong_count >= 2 and weak_count == 0:
            reliability_label = "STRONG"
        else:
            reliability_label = "MODERATE"

        is_candidate_linear = bool(compactness < 0.06 and area_ha >= 0.04)
        is_candidate_built_up = bool(
            (change_type == "surface_brightening" or change_type == "backscatter_increase")
            and compactness >= 0.12
        )
        is_candidate_water = bool(
            change_type in ("water_expansion", "water_recession")
            or (modality == "sar" and change_type == "backscatter_decrease")
        )
        is_candidate_agri = bool(
            change_type in ("vegetation_gain", "vegetation_loss")
            or (change_type == "surface_brightening" and 0.04 <= compactness <= 0.35)
        )
        is_candidate_flood = bool(
            change_type == "water_expansion"
            or (modality == "sar" and change_type == "backscatter_decrease")
        )

        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "region_id": f"CR-{len(features) + 1:04d}",
                "area_m2": round(float(area_m2), 2),
                "area_ha": round(float(area_ha), 4),
                "perimeter_m": round(float(perimeter_m), 2),
                "compactness": round(float(compactness), 4),
                "centroid_projected": [round(float(cx_proj), 2), round(float(cy_proj), 2)],
                "centroid_wgs84": [round(float(lon), 6), round(float(lat), 6)],
                "bounding_box_projected": [round(float(v), 2) for v in bbox_proj],
                "bounding_box_wgs84": bbox_wgs84,
                "mean_change_magnitude": round(float(mean_mag), 4),
                "max_change_magnitude": round(float(max_mag), 4),
                "detected_change": change_name,
                "likely_change_type": change_type,
                "change_hypothesis": hypothesis,
                "reliability_label": reliability_label,
                "candidate_linear_infrastructure": is_candidate_linear,
                "candidate_built_up": is_candidate_built_up,
                "candidate_water": is_candidate_water,
                "candidate_agriculture": is_candidate_agri,
                "candidate_flooding": is_candidate_flood,
                "evidence_factors": {
                    "area_rating": area_rating,
                    "magnitude_ratio": round(float(mag_ratio), 2),
                    "magnitude_rating": mag_rating,
                    "registration_shift_px": round(float(registration_shift_magnitude), 2),
                    "registration_rating": reg_rating,
                    "compactness_rating": comp_rating,
                    "radiometric_normalization": radiometric_method,
                    "model_agreement": cf_rating,
                },
                "disagreement_flag": disagreement,
                "spectral_metrics": metrics,
            },
        })

    features.sort(key=lambda f: f["properties"]["area_m2"], reverse=True)
    for idx, feature in enumerate(features, 1):
        feature["properties"]["region_id"] = f"CR-{idx:04d}"

    return features


def parse_gis_query(query: str) -> Dict[str, Any]:
    q = (query or "").lower()
    criteria: Dict[str, Any] = {
        "top_k": None,
        "sort_by": "area_desc",
        "min_area_ha": None,
        "max_area_ha": None,
        "change_type": None,
        "candidate_category": None,
        "reliability": None,
    }

    top_k_match = re.search(r"\b(?:top|first|show(?:\s+the)?)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", q)
    words_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    if top_k_match:
        val = top_k_match.group(1)
        criteria["top_k"] = words_to_num.get(val, int(val) if val.isdigit() else None)

    if "largest" in q or "biggest" in q or "greatest" in q:
        criteria["sort_by"] = "area_desc"
        is_singular_superlative = bool(
            re.search(r"\b(?:the|single)\s+(?:single\s+)?(?:largest|biggest|greatest)\s+(?:changed\s+)?(?:region|area|patch|cluster|zone)\b", q)
            or re.search(r"\bwhere\s+is\s+the\s+(?:single\s+)?(?:largest|biggest|greatest)\b", q)
            or "the largest changed region" in q
            or "the single largest" in q
            or "the largest region" in q
        )
        if is_singular_superlative and criteria["top_k"] is None:
            criteria["top_k"] = 1
    elif "smallest" in q:
        criteria["sort_by"] = "area_asc"
    elif "highest magnitude" in q or "most significant" in q:
        criteria["sort_by"] = "magnitude_desc"

    min_match = re.search(r"(?:greater than|more than|above|exceeding|\>)\s*([\d\.]+)\s*(?:ha|hectare|hectares)", q)
    if min_match:
        criteria["min_area_ha"] = float(min_match.group(1))

    max_match = re.search(r"(?:less than|under|below|\<)\s*([\d\.]+)\s*(?:ha|hectare|hectares)", q)
    if max_match:
        criteria["max_area_ha"] = float(max_match.group(1))

    if "vegetation loss" in q or "forest loss" in q or "deforestation" in q:
        criteria["change_type"] = "vegetation_loss"
    elif "vegetation gain" in q or "reforestation" in q:
        criteria["change_type"] = "vegetation_gain"
    elif "water" in q and ("gain" in q or "expansion" in q or "flood" in q):
        criteria["change_type"] = "water_expansion"
    elif "water" in q and ("loss" in q or "recession" in q or "shrink" in q):
        criteria["change_type"] = "water_recession"
    elif "brightening" in q:
        criteria["change_type"] = "surface_brightening"
    elif "darkening" in q:
        criteria["change_type"] = "surface_darkening"

    if any(w in q for w in ("urbanisation", "urbanization", "urban", "built-up", "built up", "construction", "building")):
        criteria["candidate_category"] = "built_up"
    elif any(w in q for w in ("agriculture", "agricultural", "farming", "farm", "crop", "crops")):
        criteria["candidate_category"] = "agriculture"
    elif any(w in q for w in ("flood", "flooding")):
        criteria["candidate_category"] = "flooding"
    elif any(w in q for w in ("road", "roads", "highway", "infrastructure")):
        criteria["candidate_category"] = "infrastructure"
    elif "water" in q and not criteria.get("change_type"):
        criteria["candidate_category"] = "water"

    if "strong" in q or "high confidence" in q or "highest confidence" in q:
        criteria["reliability"] = "STRONG"
    elif "moderate" in q:
        criteria["reliability"] = "MODERATE"
    elif "weak" in q:
        criteria["reliability"] = "WEAK"

    return criteria


def filter_and_rank_regions(
    regions: List[Dict[str, Any]],
    criteria: Dict[str, Any],
) -> List[Dict[str, Any]]:
    filtered = list(regions)

    if criteria.get("min_area_ha") is not None:
        filtered = [r for r in filtered if r.get("properties", {}).get("area_ha", 0.0) >= criteria["min_area_ha"]]

    if criteria.get("max_area_ha") is not None:
        filtered = [r for r in filtered if r.get("properties", {}).get("area_ha", 0.0) <= criteria["max_area_ha"]]

    if criteria.get("change_type"):
        matched = [r for r in filtered if r.get("properties", {}).get("likely_change_type") == criteria["change_type"]]
        if matched:
            filtered = matched

    if criteria.get("candidate_category"):
        cat = criteria["candidate_category"]
        flag_map = {
            "built_up": "candidate_built_up",
            "agriculture": "candidate_agriculture",
            "flooding": "candidate_flooding",
            "infrastructure": "candidate_linear_infrastructure",
            "water": "candidate_water",
        }
        flag = flag_map.get(cat)
        if flag:
            matched = [r for r in filtered if r.get("properties", {}).get(flag)]
            if matched:
                filtered = matched

    if criteria.get("reliability"):
        filtered = [r for r in filtered if r.get("properties", {}).get("reliability_label") == criteria["reliability"]]

    sort_by = criteria.get("sort_by", "area_desc")
    if sort_by == "area_desc":
        filtered.sort(key=lambda r: r.get("properties", {}).get("area_m2", 0.0), reverse=True)
    elif sort_by == "area_asc":
        filtered.sort(key=lambda r: r.get("properties", {}).get("area_m2", 0.0))
    elif sort_by == "magnitude_desc":
        filtered.sort(key=lambda r: r.get("properties", {}).get("mean_change_magnitude", 0.0), reverse=True)

    top_k = criteria.get("top_k")
    if top_k is not None and top_k > 0:
        filtered = filtered[:top_k]

    return filtered


def geometry_to_wgs84(
    geometry: Dict[str, Any],
    source_crs: CRS,
) -> Dict[str, Any]:
    """
    Transform a GeoJSON-compatible geometry from the analysis CRS
    into WGS84 longitude/latitude coordinates.
    """

    source_crs = CRS.from_user_input(
        source_crs
    )

    target_crs = CRS.from_epsg(
        4326
    )

    if source_crs == target_crs:
        return geometry

    transformer = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )

    def transform_point(
        point: List[float],
    ) -> List[float]:

        x, y = transformer.transform(
            point[0],
            point[1],
        )

        return [
            float(x),
            float(y),
        ]

    def transform_coordinates(
        value: Any,
    ) -> Any:

        if (
            isinstance(value, list)
            and len(value) >= 2
            and isinstance(
                value[0],
                (int, float),
            )
            and isinstance(
                value[1],
                (int, float),
            )
        ):
            return transform_point(
                value
            )

        if isinstance(value, list):
            return [
                transform_coordinates(
                    item
                )
                for item in value
            ]

        return value

    return {
        "type": geometry["type"],
        "coordinates": transform_coordinates(
            geometry["coordinates"]
        ),
    }


def features_to_wgs84(
    features: List[Dict[str, Any]],
    source_crs: CRS,
) -> List[Dict[str, Any]]:
    """
    Transform a list of generated features to WGS84.
    """

    result = []

    for feature in features:

        result.append(
            {
                "type": "Feature",
                "geometry": geometry_to_wgs84(
                    feature["geometry"],
                    source_crs,
                ),
                "properties": dict(
                    feature.get(
                        "properties",
                        {},
                    )
                ),
            }
        )

    return result


def save_geojson(
    features: List[Dict[str, Any]],
    output_path: str | Path,
    source_crs: CRS,
) -> str:
    """
    Save generated change polygons as a standard WGS84 GeoJSON file.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    wgs84_features = features_to_wgs84(
        features,
        source_crs,
    )

    collection = {
        "type": "FeatureCollection",
        "name": output_path.stem,
        "crs": {
            "type": "name",
            "properties": {
                "name": "EPSG:4326",
            },
        },
        "features": wgs84_features,
    }

    output_path.write_text(
        json.dumps(
            collection,
            indent=2,
        ),
        encoding="utf-8",
    )

    return str(output_path)


def feature_area_summary(
    features: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Summarise polygon-level change evidence.
    """

    areas = [
        float(
            feature.get(
                "properties",
                {},
            ).get(
                "area_m2",
                0.0,
            )
        )
        for feature in features
    ]

    areas = [
        area
        for area in areas
        if area > 0
    ]

    total_area = sum(
        areas
    )

    return {
        "region_count": len(
            areas
        ),
        "total_area_m2": total_area,
        "total_area_ha": (
            total_area
            / 10_000.0
        ),
        "largest_region_m2": (
            max(areas)
            if areas
            else 0.0
        ),
        "smallest_region_m2": (
            min(areas)
            if areas
            else 0.0
        ),
    }


def choose_equal_area_crs() -> CRS:
    """
    Return a global equal-area CRS suitable for general area calculations.

    EPSG:6933 is used when a local projected CRS is not already available.
    """

    return CRS.from_epsg(
        6933
    )


def validate_projected_area_crs(
    crs: CRS,
) -> bool:
    """
    Verify that an analysis CRS is projected and uses metre-like units.

    The Change Detective should never calculate area directly in
    geographic degrees.
    """

    crs = CRS.from_user_input(
        crs
    )

    if not crs.is_projected:
        return False

    axis_info = crs.axis_info

    if not axis_info:
        return False

    units = [
        (
            axis.unit_name or ""
        ).lower()
        for axis in axis_info
    ]

    return any(
        unit in (
            "metre",
            "meter",
            "metres",
            "meters",
        )
        for unit in units
    )


def calculate_overlap_fraction(
    first_bounds: Tuple[
        float,
        float,
        float,
        float,
    ],
    second_bounds: Tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    """
    Calculate intersection area divided by the smaller footprint area.

    Bounds must be in the same CRS.
    """

    first_left, first_bottom, first_right, first_top = (
        first_bounds
    )

    second_left, second_bottom, second_right, second_top = (
        second_bounds
    )

    left = max(
        first_left,
        second_left,
    )

    bottom = max(
        first_bottom,
        second_bottom,
    )

    right = min(
        first_right,
        second_right,
    )

    top = min(
        first_top,
        second_top,
    )

    if (
        right <= left
        or top <= bottom
    ):
        return 0.0

    intersection_area = (
        right - left
    ) * (
        top - bottom
    )

    first_area = max(
        0.0,
        first_right - first_left,
    ) * max(
        0.0,
        first_top - first_bottom,
    )

    second_area = max(
        0.0,
        second_right - second_left,
    ) * max(
        0.0,
        second_top - second_bottom,
    )

    denominator = min(
        first_area,
        second_area,
    )

    if denominator <= 0:
        return 0.0

    return float(
        intersection_area
        / denominator
    )