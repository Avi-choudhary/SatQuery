from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pyproj import CRS, Transformer
from rasterio.features import shapes
from rasterio.transform import Affine


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