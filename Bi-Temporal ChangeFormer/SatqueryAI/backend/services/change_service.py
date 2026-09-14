from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from pyproj import CRS, Transformer

# change_service is loaded by path via importlib from the root backend, so a
# plain `from . import irmad` has no package context to resolve against.
import importlib.util as _importlib_util

_IRMAD_PATH = Path(__file__).resolve().parent / "irmad.py"
try:
    _spec = _importlib_util.spec_from_file_location("satquery_irmad", _IRMAD_PATH)
    irmad_module = _importlib_util.module_from_spec(_spec)
    import sys as _sys
    _sys.modules["satquery_irmad"] = irmad_module  # dataclass needs this
    _spec.loader.exec_module(irmad_module)
except Exception as _irmad_exc:  # pragma: no cover - optional dependency path
    irmad_module = None
    _IRMAD_IMPORT_ERROR = _irmad_exc
else:
    _IRMAD_IMPORT_ERROR = None


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "change_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUPPORTED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
}

# Maximum residual translation that we consider acceptable without warning.
MAX_REGISTRATION_SHIFT_PIXELS = 3.0

# Very small regions are normally noise rather than meaningful change.
MIN_REGION_PIXELS = 9

# Prevent pathological memory usage on extremely large rasters.
MAX_ANALYSIS_PIXELS = 20_000_000

# ChangeFormer is used as a candidate generator for compatible optical
# general-change queries. Extremely large candidate masks are treated as
# suspicious model output and are verified using the deterministic path.
MAX_CHANGEFORMER_CANDIDATE_FRACTION = 0.80

# Numerical tolerance used to identify a degenerate/flat change magnitude
# distribution. A flat distribution must never become an all-image mask.
CHANGE_SIGNAL_EPSILON = 1e-6


# =============================================================================
# Generic helpers
# =============================================================================

def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)

        if math.isfinite(result):
            return result

    except (TypeError, ValueError):
        pass

    return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_query(query: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        (query or "").strip().lower(),
    )


def _query_feature(query: str) -> str:
    """
    Decide whether the query specifically asks about a broad feature.

    This does NOT assume that the corresponding spectral index exists.
    Band availability is checked later.
    """

    text = _normalise_query(query)

    if any(
        phrase in text
        for phrase in (
            "water body",
            "water bodies",
            "water area",
            "water extent",
            "lake",
            "lakes",
            "river",
            "rivers",
            "reservoir",
            "reservoirs",
        )
    ):
        return "water"

    if any(
        phrase in text
        for phrase in (
            "built-up",
            "built up",
            "builtup",
            "urban area",
            "urban areas",
            "construction",
            "buildings",
            "building",
        )
    ):
        return "built_up"

    if any(
        phrase in text
        for phrase in (
            "vegetation",
            "vegetated",
            "green cover",
            "greenery",
            "forest",
            "forested",
            "crop",
            "crops",
            "agriculture",
            "agricultural",
        )
    ):
        return "vegetation"

    if any(
        phrase in text
        for phrase in (
            "road",
            "roads",
        )
    ):
        return "road"

    if any(
        phrase in text
        for phrase in (
            "waste",
            "dump",
            "landfill",
            "garbage",
        )
    ):
        return "waste"

    return "general"


def _detect_modality(
    dataset: rasterio.io.DatasetReader,
) -> str:
    """
    Conservative modality identification.

    Optical imagery is assumed when ordinary reflectance-like band counts
    are present. SAR is detected from metadata where available.
    """

    tags = dataset.tags()

    text = " ".join(
        [
            str(dataset.driver),
            str(dataset.count),
            str(tags),
            str(dataset.descriptions),
            str(dataset.name),
        ]
    ).lower()

    sar_keywords = (
        "sar",
        "sentinel-1",
        "risat",
        "radar",
        "sigma0",
        "gamma0",
        "backscatter",
        "vv",
        "vh",
    )

    if any(keyword in text for keyword in sar_keywords):
        return "sar"

    return "optical"


def _has_georeferencing(
    dataset: rasterio.io.DatasetReader,
) -> bool:
    """
    CRS and a real affine geotransform are mandatory for the geographic
    Change Detective workflow.
    """

    if dataset.crs is None:
        return False

    transform = dataset.transform

    if transform is None:
        return False

    if transform == Affine.identity():
        return False

    if (
        transform.a == 0
        or transform.e == 0
    ):
        return False

    return True


def _synthesize_georeferencing(
    dataset: rasterio.io.DatasetReader,
) -> Tuple[CRS, Affine]:
    """
    Create a synthetic CRS and transform for non-georeferenced images
    (e.g. PNG/JPG photos) so that the change detection pipeline can
    proceed.  The synthetic transform maps pixel coordinates into a
    small 10 m GSD area around a default centre point (Bengaluru).
    """
    # 10 m pixel size in EPSG:3857 (Web Mercator, metres)
    pixel_size = 10.0
    # Arbitrary origin – roughly Bengaluru in EPSG:3857 metres
    origin_x = 8_640_000.0
    origin_y = 1_460_000.0
    synthetic_crs = CRS.from_epsg(3857)
    synthetic_transform = Affine(
        pixel_size,
        0.0,
        origin_x,
        0.0,
        -pixel_size,
        origin_y + dataset.height * pixel_size,
    )
    return synthetic_crs, synthetic_transform


def _valid_transform(
    transform: Affine,
) -> bool:
    return (
        transform is not None
        and transform.a != 0
        and transform.e != 0
    )


def _resolution(
    transform: Affine,
) -> Tuple[float, float]:
    return (
        abs(float(transform.a)),
        abs(float(transform.e)),
    )


def _bounds_intersection(
    a: rasterio.coords.BoundingBox,
    b: rasterio.coords.BoundingBox,
) -> Optional[Tuple[float, float, float, float]]:
    left = max(a.left, b.left)
    bottom = max(a.bottom, b.bottom)
    right = min(a.right, b.right)
    top = min(a.top, b.top)

    if right <= left or top <= bottom:
        return None

    return (
        left,
        bottom,
        right,
        top,
    )


def _intersection_fraction(
    a: rasterio.coords.BoundingBox,
    b: rasterio.coords.BoundingBox,
) -> float:
    intersection = _bounds_intersection(a, b)

    if intersection is None:
        return 0.0

    left, bottom, right, top = intersection

    intersection_area = (
        (right - left)
        * (top - bottom)
    )

    area_a = (
        max(0.0, a.right - a.left)
        * max(0.0, a.top - a.bottom)
    )

    area_b = (
        max(0.0, b.right - b.left)
        * max(0.0, b.top - b.bottom)
    )

    denominator = min(area_a, area_b)

    if denominator <= 0:
        return 0.0

    return intersection_area / denominator


# =============================================================================
# Band interpretation
# =============================================================================

def _band_names(
    dataset: rasterio.io.DatasetReader,
) -> List[str]:
    names: List[str] = []

    descriptions = dataset.descriptions

    for index in range(dataset.count):
        description = descriptions[index]

        if description:
            names.append(
                str(description).strip().lower()
            )
        else:
            names.append(
                f"band_{index + 1}"
            )

    return names


def _find_band(
    names: List[str],
    candidates: Tuple[str, ...],
) -> Optional[int]:
    for candidate in candidates:
        candidate = candidate.lower()

        for index, name in enumerate(names):
            clean = name.replace(" ", "").replace("_", "")

            if candidate.replace("_", "") in clean:
                return index + 1

    return None


def _find_spectral_bands(
    dataset: rasterio.io.DatasetReader,
) -> Dict[str, Optional[int]]:
    names = _band_names(dataset)

    red = _find_band(
        names,
        ("red", "b4"),
    )

    green = _find_band(
        names,
        ("green", "b3"),
    )

    blue = _find_band(
        names,
        ("blue", "b2"),
    )

    nir = _find_band(
        names,
        (
            "nir",
            "b8",
            "b8a",
        ),
    )

    swir1 = _find_band(
        names,
        (
            "swir1",
            "swir",
            "b11",
        ),
    )

    return {
        "blue": blue,
        "green": green,
        "red": red,
        "nir": nir,
        "swir1": swir1,
    }


# =============================================================================
# Raster preprocessing
# =============================================================================

def _read_valid_mask(
    dataset: rasterio.io.DatasetReader,
) -> np.ndarray:
    """
    Read the raster's native mask.

    We deliberately do not invent a cloud mask when the source does not
    contain one.
    """

    mask = dataset.dataset_mask()

    return mask > 0


def _read_array(
    dataset: rasterio.io.DatasetReader,
    indexes: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if indexes is None:
        indexes = list(
            range(
                1,
                dataset.count + 1,
            )
        )

    array = dataset.read(
        indexes=indexes,
        masked=True,
    )

    if np.ma.isMaskedArray(array):
        valid = ~np.any(
            np.ma.getmaskarray(array),
            axis=0,
        )

        data = np.asarray(
            array.filled(np.nan),
            dtype=np.float32,
        )

    else:
        data = np.asarray(
            array,
            dtype=np.float32,
        )

        valid = np.all(
            np.isfinite(data),
            axis=0,
        )

    dataset_mask = _read_valid_mask(dataset)

    valid &= dataset_mask

    valid &= np.all(
        np.isfinite(data),
        axis=0,
    )

    return data, valid


def _robust_scale(
    array: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Per-band robust scaling.

    Percentile normalization is used only within each modality image.
    SAR and optical data never share the same normalization operation.
    """

    output = np.zeros_like(
        array,
        dtype=np.float32,
    )

    finite = (
        valid
        & np.isfinite(array)
    )

    values = array[finite]

    if values.size < 10:
        return output

    low, high = np.percentile(
        values,
        [2, 98],
    )

    if not np.isfinite(low) or not np.isfinite(high):
        return output

    if high <= low:
        output[finite] = 0.0
        return output

    output[finite] = np.clip(
        (
            array[finite] - low
        )
        / (
            high - low
        ),
        0.0,
        1.0,
    )

    return output


# =============================================================================
# Common projected analysis grid
# =============================================================================

def _choose_analysis_crs(
    first: rasterio.io.DatasetReader,
    second: rasterio.io.DatasetReader,
) -> CRS:
    """
    Prefer an equal-area CRS for area reporting.

    If both images already share a projected CRS, keep it.
    Otherwise use EPSG:6933 (World Equidistant Cylindrical / equal-area
    global analysis CRS).
    """

    first_crs = CRS.from_user_input(first.crs)
    second_crs = CRS.from_user_input(second.crs)

    if (
        first_crs == second_crs
        and first_crs.is_projected
    ):
        return first_crs

    # EPSG:6933 is a global equal-area projected CRS.
    return CRS.from_epsg(6933)


def _choose_resolution(
    first: rasterio.io.DatasetReader,
    second: rasterio.io.DatasetReader,
) -> Tuple[float, float]:
    """
    Avoid blindly upsampling the coarser source.

    We use the coarser native ground sampling distance.
    """

    r1x, r1y = _resolution(first.transform)
    r2x, r2y = _resolution(second.transform)

    return (
        max(r1x, r2x),
        max(r1y, r2y),
    )


def _make_common_grid(
    first: rasterio.io.DatasetReader,
    second: rasterio.io.DatasetReader,
) -> Dict[str, Any]:

    analysis_crs = _choose_analysis_crs(
        first,
        second,
    )

    resolution_x, resolution_y = _choose_resolution(
        first,
        second,
    )

    first_transform, first_width, first_height = (
        calculate_default_transform(
            first.crs,
            analysis_crs,
            first.width,
            first.height,
            *first.bounds,
            resolution=(
                resolution_x,
                resolution_y,
            ),
        )
    )

    second_transform, second_width, second_height = (
        calculate_default_transform(
            second.crs,
            analysis_crs,
            second.width,
            second.height,
            *second.bounds,
            resolution=(
                resolution_x,
                resolution_y,
            ),
        )
    )

    def projected_bounds(
        transform: Affine,
        width: int,
        height: int,
    ) -> Tuple[float, float, float, float]:

        corners = [
            transform * (0, 0),
            transform * (width, 0),
            transform * (0, height),
            transform * (width, height),
        ]

        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]

        return (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

    first_bounds = projected_bounds(
        first_transform,
        first_width,
        first_height,
    )

    second_bounds = projected_bounds(
        second_transform,
        second_width,
        second_height,
    )

    left = max(
        first_bounds[0],
        second_bounds[0],
    )

    bottom = max(
        first_bounds[1],
        second_bounds[1],
    )

    right = min(
        first_bounds[2],
        second_bounds[2],
    )

    top = min(
        first_bounds[3],
        second_bounds[3],
    )

    if right <= left or top <= bottom:
        raise ValueError(
            "The two images do not have a geographic overlap."
        )

    width = int(
        math.ceil(
            (right - left)
            / resolution_x
        )
    )

    height = int(
        math.ceil(
            (top - bottom)
            / resolution_y
        )
    )

    if width <= 0 or height <= 0:
        raise ValueError(
            "The common analysis grid is empty."
        )

    if width * height > MAX_ANALYSIS_PIXELS:
        raise ValueError(
            "The common analysis area is too large for safe in-memory "
            "processing. Use a smaller scene or tile the analysis."
        )

    transform = Affine(
        resolution_x,
        0,
        left,
        0,
        -resolution_y,
        top,
    )

    return {
        "crs": analysis_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "resolution_x": resolution_x,
        "resolution_y": resolution_y,
        "bounds": (
            left,
            bottom,
            right,
            top,
        ),
    }


def _reproject_to_grid(
    dataset: rasterio.io.DatasetReader,
    indexes: List[int],
    grid: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:

    destination = np.full(
        (
            len(indexes),
            grid["height"],
            grid["width"],
        ),
        np.nan,
        dtype=np.float32,
    )

    destination_valid = np.zeros(
        (
            grid["height"],
            grid["width"],
        ),
        dtype=bool,
    )

    for output_index, source_index in enumerate(indexes):

        source = dataset.read(
            source_index,
            masked=True,
        )

        if np.ma.isMaskedArray(source):
          source_valid = (
            ~np.ma.getmaskarray(source)
          )

          source_array = np.asarray(
            source.data,
            dtype=np.float32,
          )

          source_array[~source_valid] = np.nan

        else:
            source_array = np.asarray(
                source,
                dtype=np.float32,
            )

            source_valid = np.isfinite(
                source_array
            )

        source_valid &= np.isfinite(
            source_array
        )

        reproject(
            source=source_array,
            destination=destination[output_index],
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        valid_float = np.zeros(
            (
                grid["height"],
                grid["width"],
            ),
            dtype=np.float32,
        )

        reproject(
            source=source_valid.astype(np.float32),
            destination=valid_float,
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        )

        destination_valid |= (
            valid_float > 0.5
        )

    destination_valid &= np.all(
        np.isfinite(destination),
        axis=0,
    )

    return (
        destination,
        destination_valid,
    )


# =============================================================================
# Residual registration check
# =============================================================================

def _estimate_translation(
    first: np.ndarray,
    second: np.ndarray,
    valid: np.ndarray,
) -> Tuple[float, float]:
    """
    Estimate residual integer-pixel translation using FFT phase correlation.

    This is a diagnostic, not an uncontrolled geometric correction.
    """

    if first.ndim == 3:
        first_image = np.mean(
            first,
            axis=0,
        )
    else:
        first_image = first

    if second.ndim == 3:
        second_image = np.mean(
            second,
            axis=0,
        )
    else:
        second_image = second

    first_image = np.nan_to_num(
        first_image,
        nan=0.0,
    )

    second_image = np.nan_to_num(
        second_image,
        nan=0.0,
    )

    first_image *= valid
    second_image *= valid

    height, width = first_image.shape

    if (
        height < 16
        or width < 16
    ):
        return 0.0, 0.0

    # Downsample large scenes to keep FFT memory bounded.
    scale = max(
        1,
        int(
            max(height, width)
            / 1024
        ),
    )

    first_small = first_image[
        ::scale,
        ::scale,
    ]

    second_small = second_image[
        ::scale,
        ::scale,
    ]

    first_small -= np.mean(
        first_small
    )

    second_small -= np.mean(
        second_small
    )

    fft_a = np.fft.fft2(
        first_small
    )

    fft_b = np.fft.fft2(
        second_small
    )

    cross_power = (
        fft_a
        * np.conj(fft_b)
    )

    denominator = np.abs(
        cross_power
    )

    denominator[
        denominator < 1e-12
    ] = 1e-12

    cross_power /= denominator

    correlation = np.fft.ifft2(
        cross_power
    )

    magnitude = np.abs(
        correlation
    )

    peak_y, peak_x = np.unravel_index(
        np.argmax(magnitude),
        magnitude.shape,
    )

    small_height, small_width = (
        magnitude.shape
    )

    if peak_y > small_height // 2:
        peak_y -= small_height

    if peak_x > small_width // 2:
        peak_x -= small_width

    shift_y = float(
        peak_y * scale
    )

    shift_x = float(
        peak_x * scale
    )

    return (
        shift_x,
        shift_y,
    )


# =============================================================================
# Change magnitude methods
# =============================================================================

def _cva_change(first, second, valid):
    """
    Generic optical temporal change magnitude.

    Uses symmetric relative difference instead of independently
    normalizing T1 and T2. This preserves genuine temporal
    radiometric/spatial differences between the two acquisitions.
    """
    epsilon = 1e-6

    first = first.astype(np.float32, copy=False)
    second = second.astype(np.float32, copy=False)

    difference = second - first

    denominator = (
        np.abs(second) +
        np.abs(first) +
        epsilon
    )

    relative_difference = np.abs(difference) / denominator

    magnitude = np.sqrt(
        np.sum(relative_difference ** 2, axis=0)
    )

    magnitude[~valid] = np.nan

    return magnitude.astype(np.float32)


def _sar_log_ratio(
    first: np.ndarray,
    second: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    SAR change magnitude using log-ratio style differencing.

    SAR intensity is clipped to a positive floor before logarithms.
    """

    first_scaled = np.zeros_like(
        first,
        dtype=np.float32,
    )

    second_scaled = np.zeros_like(
        second,
        dtype=np.float32,
    )

    for band in range(
        first.shape[0]
    ):
        a = np.abs(
            first[band]
        )

        b = np.abs(
            second[band]
        )

        a = np.maximum(
            a,
            1e-6,
        )

        b = np.maximum(
            b,
            1e-6,
        )

        first_scaled[band] = np.log(
            a
        )

        second_scaled[band] = np.log(
            b
        )

    difference = (
        second_scaled
        - first_scaled
    )

    magnitude = np.sqrt(
        np.sum(
            difference ** 2,
            axis=0,
        )
    )

    magnitude[~valid] = np.nan

    return magnitude.astype(
        np.float32
    )


def _normalised_difference(
    first: np.ndarray,
    second: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Robust feature-space difference for RGB/other optical imagery.
    """

    return _cva_change(
        first,
        second,
        valid,
    )


# =============================================================================
# Query-specific feature analysis
# =============================================================================

def _index(
    data: np.ndarray,
    numerator_band_a: int,
    numerator_band_b: int,
    valid: np.ndarray,
) -> np.ndarray:

    a = data[
        numerator_band_a - 1
    ]

    b = data[
        numerator_band_b - 1
    ]

    denominator = a + b

    result = np.full(
        a.shape,
        np.nan,
        dtype=np.float32,
    )

    safe = (
        valid
        & np.isfinite(a)
        & np.isfinite(b)
        & (
            np.abs(denominator)
            > 1e-6
        )
    )

    result[safe] = (
        (
            a[safe]
            - b[safe]
        )
        /
        denominator[safe]
    )

    return result


def _query_specific_magnitude(
    feature: str,
    first_data: np.ndarray,
    second_data: np.ndarray,
    first_dataset: rasterio.io.DatasetReader,
    second_dataset: rasterio.io.DatasetReader,
    valid: np.ndarray,
) -> Tuple[Optional[np.ndarray], str]:

    if feature == "general":
        return None, "general_change"

    if _detect_modality(first_dataset) != "optical":
        return None, "general_change"

    first_bands = _find_spectral_bands(
        first_dataset
    )

    second_bands = _find_spectral_bands(
        second_dataset
    )

    if feature == "vegetation":

        if (
            first_bands["nir"] is None
            or first_bands["red"] is None
            or second_bands["nir"] is None
            or second_bands["red"] is None
        ):
            return None, "general_change"

        first_index = _index(
            first_data,
            first_bands["nir"],
            first_bands["red"],
            valid,
        )

        second_index = _index(
            second_data,
            second_bands["nir"],
            second_bands["red"],
            valid,
        )

        magnitude = np.abs(
            second_index
            - first_index
        )

        magnitude[~valid] = np.nan

        return (
            magnitude.astype(
                np.float32
            ),
            "ndvi_difference",
        )

    if feature == "built_up":

        if (
            first_bands["nir"] is None
            or first_bands["swir1"] is None
            or second_bands["nir"] is None
            or second_bands["swir1"] is None
        ):
            return None, "general_change"

        first_index = _index(
            first_data,
            first_bands["swir1"],
            first_bands["nir"],
            valid,
        )

        second_index = _index(
            second_data,
            second_bands["swir1"],
            second_bands["nir"],
            valid,
        )

        magnitude = np.abs(
            second_index
            - first_index
        )

        magnitude[~valid] = np.nan

        return (
            magnitude.astype(
                np.float32
            ),
            "ndbi_difference",
        )

    if feature == "water":

        if (
            first_bands["green"] is None
            or first_bands["nir"] is None
            or second_bands["green"] is None
            or second_bands["nir"] is None
        ):
            return None, "general_change"

        first_index = _index(
            first_data,
            first_bands["green"],
            first_bands["nir"],
            valid,
        )

        second_index = _index(
            second_data,
            second_bands["green"],
            second_bands["nir"],
            valid,
        )

        magnitude = np.abs(
            second_index
            - first_index
        )

        magnitude[~valid] = np.nan

        return (
            magnitude.astype(
                np.float32
            ),
            "ndwi_difference",
        )

    # Roads and waste currently do not receive an invented spectral index.
    return None, "general_change"


# =============================================================================
# Thresholding
# =============================================================================

def _otsu_threshold(
    values: np.ndarray,
) -> Optional[float]:

    values = values[
        np.isfinite(values)
    ]

    if values.size < 32:
        return None

    low, high = np.percentile(
        values,
        [1, 99],
    )

    if not np.isfinite(low) or not np.isfinite(high):
        return None

    if high <= low:
        return None

    clipped = np.clip(
        values,
        low,
        high,
    )

    histogram, edges = np.histogram(
        clipped,
        bins=256,
        range=(low, high),
    )

    histogram = histogram.astype(
        np.float64
    )

    total = histogram.sum()

    if total <= 0:
        return None

    probabilities = (
        histogram
        / total
    )

    centres = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    omega = np.cumsum(
        probabilities
    )

    mu = np.cumsum(
        probabilities
        * centres
    )

    total_mean = mu[-1]

    denominator = (
        omega
        * (
            1.0
            - omega
        )
    )

    numerator = (
        total_mean
        * omega
        - mu
    ) ** 2

    score = np.zeros_like(
        numerator
    )

    safe = denominator > 1e-12

    score[safe] = (
        numerator[safe]
        / denominator[safe]
    )

    if not np.any(safe):
        return None

    index = int(
        np.argmax(score)
    )

    threshold = float(
        centres[index]
    )

    # If the threshold is extremely close to either distribution edge,
    # Otsu is not giving us a useful split.
    dynamic_range = high - low

    if (
        threshold
        <= low
        + 0.01 * dynamic_range
    ):
        return None

    if (
        threshold
        >= high
        - 0.01 * dynamic_range
    ):
        return None

    return threshold


def _threshold_change(
    magnitude: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, float, str]:
    """
    Convert a continuous change-magnitude raster into a binary mask.

    Safety rules:
    - Never threshold a flat/degenerate distribution at zero.
    - Never turn an all-zero magnitude raster into an all-change mask.
    - Prefer Otsu when it provides a meaningful split.
    - Use the 95th percentile only as a fallback.
    - If the resulting mask is implausibly large, use a stricter 99th
      percentile safety fallback.
    """

    values = magnitude[
        valid
        & np.isfinite(magnitude)
    ]

    if values.size == 0:
        raise ValueError(
            "No valid change-analysis pixels remain."
        )

    if values.size < 32:
        return (
            np.zeros_like(valid, dtype=bool),
            float("nan"),
            "no_reliable_change_signal",
        )

    value_min = float(np.min(values))
    value_max = float(np.max(values))

    if (
        not np.isfinite(value_min)
        or not np.isfinite(value_max)
    ):
        return (
            np.zeros_like(valid, dtype=bool),
            float("nan"),
            "no_reliable_change_signal",
        )

    dynamic_range = value_max - value_min

    # Critical guard:
    # if all pixels have essentially the same magnitude, thresholding at
    # zero/that constant would incorrectly classify the whole scene.
    tolerance = max(
        CHANGE_SIGNAL_EPSILON,
        abs(value_max) * 1e-6,
    )

    if (
        dynamic_range <= tolerance
        or value_max <= CHANGE_SIGNAL_EPSILON
    ):
        return (
            np.zeros_like(valid, dtype=bool),
            float("nan"),
            "no_reliable_change_signal",
        )

    threshold = _otsu_threshold(values)
    method = "otsu"

    if threshold is None:
        threshold = float(
            np.percentile(
                values,
                95,
            )
        )
        method = "percentile_95"

    if (
        not np.isfinite(threshold)
        or threshold <= value_min
    ):
        threshold = float(
            np.percentile(
                values,
                95,
            )
        )
        method = "percentile_95"

    if (
        not np.isfinite(threshold)
        or threshold <= CHANGE_SIGNAL_EPSILON
    ):
        return (
            np.zeros_like(valid, dtype=bool),
            float("nan"),
            "no_reliable_change_signal",
        )

    changed = (
        magnitude >= threshold
    )

    changed &= valid

    valid_pixels = int(
        np.count_nonzero(valid)
    )

    changed_pixels = int(
        np.count_nonzero(changed)
    )

    fraction = (
        changed_pixels
        / max(1, valid_pixels)
    )

    # A 95th-percentile/Otsu result covering most of the entire scene is
    # usually a sign of a weak/poorly separated signal. Tighten it before
    # accepting an enormous change mask.
    if fraction > MAX_CHANGEFORMER_CANDIDATE_FRACTION:
        safer_threshold = float(
            np.percentile(
                values,
                99,
            )
        )

        safer_changed = (
            magnitude >= safer_threshold
        )

        safer_changed &= valid

        safer_fraction = (
            np.count_nonzero(safer_changed)
            / max(1, valid_pixels)
        )

        if (
            np.isfinite(safer_threshold)
            and safer_threshold > CHANGE_SIGNAL_EPSILON
            and safer_fraction <= MAX_CHANGEFORMER_CANDIDATE_FRACTION
        ):
            changed = safer_changed
            threshold = safer_threshold
            method = "percentile_99_safety"
        else:
            return (
                np.zeros_like(valid, dtype=bool),
                float("nan"),
                "no_reliable_change_signal",
            )

    return (
        changed.astype(bool),
        float(threshold),
        method,
    )


# =============================================================================
# Mask cleaning
# =============================================================================

def _clean_mask(
    mask: np.ndarray,
) -> np.ndarray:

    try:
        from scipy import ndimage

        # Remove isolated one-pixel noise.
        cleaned = ndimage.binary_opening(
            mask,
            structure=np.ones(
                (3, 3),
                dtype=bool,
            ),
        )

        # Fill tiny holes inside changed regions.
        cleaned = ndimage.binary_closing(
            cleaned,
            structure=np.ones(
                (3, 3),
                dtype=bool,
        )

        )

        labels, count = ndimage.label(
            cleaned,
            structure=np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
        )

        if count == 0:
            return cleaned.astype(bool)

        sizes = np.bincount(
            labels.ravel()
        )

        keep = sizes >= MIN_REGION_PIXELS

        keep[0] = False

        cleaned = keep[
            labels
        ]

        return cleaned.astype(bool)

    except ImportError:
        # Basic fallback if scipy is unavailable.
        return mask.astype(bool)


# =============================================================================
# Polygonization / GeoJSON
# =============================================================================

def _polygonize(
    mask: np.ndarray,
    transform: Affine,
    crs: CRS,
) -> List[Dict[str, Any]]:

    features: List[Dict[str, Any]] = []

    for geometry, value in shapes(
        mask.astype(np.uint8),
        mask=mask,
        transform=transform,
    ):

        if not value:
            continue

        # Compute area in the projected analysis CRS.
        area_m2 = _geometry_area_m2(
            geometry,
            crs,
        )

        if area_m2 <= 0:
            continue

        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "area_m2": float(area_m2),
                    "area_ha": float(
                        area_m2 / 10_000.0
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


def _geometry_area_m2(
    geometry: Dict[str, Any],
    crs: CRS,
) -> float:
    """
    Geometry is already in the projected analysis CRS.

    For a projected CRS whose units are metres, shoelace-based area is
    sufficient for the generated raster polygons.
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

        rings = coordinates

        if not rings:
            return 0.0

        outer = _ring_area(
            rings[0]
        )

        holes = sum(
            _ring_area(ring)
            for ring in rings[1:]
        )

        return max(
            0.0,
            outer - holes,
        )

    if geometry_type == "MultiPolygon":

        total = 0.0

        for polygon in coordinates:

            if not polygon:
                continue

            outer = _ring_area(
                polygon[0]
            )

            holes = sum(
                _ring_area(ring)
                for ring in polygon[1:]
            )

            total += max(
                0.0,
                outer - holes,
            )

        return total

    return 0.0


def _ring_area(
    ring: List[List[float]],
) -> float:

    if len(ring) < 3:
        return 0.0

    area = 0.0

    for index in range(
        len(ring) - 1
    ):
        x1, y1 = ring[index]
        x2, y2 = ring[index + 1]

        area += (
            x1 * y2
            - x2 * y1
        )

    return abs(area) / 2.0


def _transform_geojson_to_wgs84(
    geometry: Dict[str, Any],
    source_crs: CRS,
) -> Dict[str, Any]:

    if source_crs.to_epsg() == 4326:
        return geometry

    transformer = Transformer.from_crs(
        source_crs,
        CRS.from_epsg(4326),
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

    def transform_nested(
        value: Any,
    ) -> Any:

        if (
            isinstance(value, (list, tuple))
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

        if isinstance(value, (list, tuple)):
            return [
                transform_nested(item)
                for item in value
            ]

        return value

    return {
        "type": geometry["type"],
        "coordinates": transform_nested(
            geometry["coordinates"]
        ),
    }


def _save_geojson(
    features: List[Dict[str, Any]],
    source_crs: CRS,
    output_path: Path,
) -> Dict[str, Any]:

    wgs84_features = []

    for feature in features:

        geometry = (
            _transform_geojson_to_wgs84(
                feature["geometry"],
                source_crs,
            )
        )

        props = dict(feature.get("properties", {}))
        area_ha = props.get("area_ha", 0.0)
        props.setdefault("label", f"Detected Change ({area_ha:.2f} ha)" if area_ha else "Detected Change")
        props.setdefault("fillColor", "#ff3b30")
        props.setdefault("strokeColor", "#ff3b30")
        props.setdefault("color", "#ff3b30")
        props.setdefault("fillOpacity", 0.32)
        props.setdefault("type", "bi-temporal-change")

        wgs84_features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": props,
            }
        )

    collection = {
        "type": "FeatureCollection",
        "features": wgs84_features,
    }

    output_path.write_text(
        json.dumps(
            collection,
            indent=2,
        ),
        encoding="utf-8",
    )

    return collection


# =============================================================================
# Evidence raster
# =============================================================================

def _save_mask_web_overlay(
    changed: np.ndarray,
    valid: np.ndarray,
    transform: Affine,
    crs: CRS,
    output_path: Path,
) -> Optional[Dict[str, Any]]:
    """
    Write the change mask as an RGBA PNG warped to Web Mercator.

    The pipeline already saves a GeoTIFF mask, but a browser cannot draw one.
    Warping to EPSG:3857 here means the PNG can be handed straight to a slippy
    map as an image overlay and land on the right ground at every zoom, which
    stretching a native-CRS render across a lat/lon box does not.

    Returns the placement metadata, or None if the warp fails.
    """
    try:
        from PIL import Image

        height, width = changed.shape

        # Red where changed, transparent everywhere else — including invalid
        # pixels, so the nodata skirt never paints over the basemap.
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        hit = changed & valid
        rgba[..., 0][hit] = 255
        rgba[..., 1][hit] = 59
        rgba[..., 2][hit] = 48
        rgba[..., 3][hit] = 190

        dst_transform, dst_width, dst_height = calculate_default_transform(
            crs, "EPSG:3857", width, height, *rasterio.transform.array_bounds(
                height, width, transform
            )
        )

        warped = np.zeros((4, dst_height, dst_width), dtype=np.uint8)
        for band in range(4):
            reproject(
                source=rgba[..., band],
                destination=warped[band],
                src_transform=transform,
                src_crs=crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:3857",
                resampling=Resampling.nearest,
            )

        Image.fromarray(
            np.transpose(warped, (1, 2, 0)), mode="RGBA"
        ).save(str(output_path), "PNG")

        left, top = dst_transform * (0, 0)
        right, bottom = dst_transform * (dst_width, dst_height)
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        min_lon, min_lat = transformer.transform(left, bottom)
        max_lon, max_lat = transformer.transform(right, top)

        return {
            "wgs84_bounds": [
                round(float(min_lon), 8),
                round(float(min_lat), 8),
                round(float(max_lon), 8),
                round(float(max_lat), 8),
            ],
            "width": int(dst_width),
            "height": int(dst_height),
        }

    except Exception as exc:
        print(f"[Change overlay warning]: {type(exc).__name__}: {exc}")
        return None


def _save_mask_raster(
    mask: np.ndarray,
    transform: Affine,
    crs: CRS,
    output_path: Path,
) -> None:

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=0,
        compress="deflate",
    ) as dst:

        dst.write(
            mask.astype(
                np.uint8
            ),
            1,
        )


def _save_magnitude_raster(
    magnitude: np.ndarray,
    transform: Affine,
    crs: CRS,
    output_path: Path,
) -> None:

    safe = np.nan_to_num(
        magnitude,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=safe.shape[0],
        width=safe.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=0,
        compress="deflate",
    ) as dst:

        dst.write(
            safe.astype(
                np.float32
            ),
            1,
        )


# =============================================================================
# Optional ChangeFormer
# =============================================================================

def _try_changeformer(
    first_data: np.ndarray,
    second_data: np.ndarray,
    valid: np.ndarray,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Run ChangeFormerV6 on the already co-registered common analysis grid.

    ChangeFormer expects 3-channel RGB inputs. The common-grid arrays are
    converted to RGB, tiled into 256x256 patches, and the resulting binary
    predictions are stitched back to the common analysis grid.
    """

    try:
        from services.changeformer_service import run_changeformer_arrays
    except Exception as exc:
        return (
            None,
            f"CHANGEFORMER_IMPORT_FAILED: {type(exc).__name__}: {exc}",
        )

    try:
        result = run_changeformer_arrays(
            first_data=first_data,
            second_data=second_data,
            valid=valid,
        )

        mask = result.get("mask")

        if mask is None:
            return (
                None,
                "CHANGEFORMER_NO_PREDICTION",
            )

        mask = np.asarray(mask)

        if mask.shape != valid.shape:
            return (
                None,
                "CHANGEFORMER_INVALID_MASK_SHAPE: "
                f"{mask.shape} != {valid.shape}",
            )

        mask = (
            (mask > 0)
            & valid
        )

        return (
            mask,
            (
                "CHANGEFORMER_EXECUTED: "
                f"{result.get('tiles_processed', 0)} tiles, "
                f"change_fraction="
                f"{result.get('change_fraction', 0.0):.4f}"
            ),
        )

    except Exception as exc:
        return (
            None,
            f"CHANGEFORMER_INFERENCE_FAILED: "
            f"{type(exc).__name__}: {exc}",
        )
# =============================================================================
# Core analysis
# =============================================================================

def _analyse_pair(
    query: str,
    first_path: str,
    second_path: str,
    tracer: Any,
) -> Tuple[str, List[Any]]:

    first_file = Path(
        first_path
    )

    second_file = Path(
        second_path
    )

    if first_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{first_file.suffix}'. "
            f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if second_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{second_file.suffix}'. "
            f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if not first_file.exists():
        raise ValueError(
            f"Before image not found: {first_file}"
        )

    if not second_file.exists():
        raise ValueError(
            f"After image not found: {second_file}"
        )

    # -------------------------------------------------------------------------
    # Open datasets
    # -------------------------------------------------------------------------

    with rasterio.open(
        first_file
    ) as first_dataset, rasterio.open(
        second_file
    ) as second_dataset:

        tracer.append_log(
            "step 3: opened before/after raster datasets"
        )

        # ---------------------------------------------------------------------
        # Georeferencing: synthesize if missing (PNG / JPG photos)
        # ---------------------------------------------------------------------

        _synth_crs_1: Optional[CRS] = None
        _synth_transform_1: Optional[Affine] = None
        _synth_crs_2: Optional[CRS] = None
        _synth_transform_2: Optional[Affine] = None

        if not _has_georeferencing(first_dataset):
            _synth_crs_1, _synth_transform_1 = _synthesize_georeferencing(
                first_dataset
            )
            tracer.append_log(
                "before image has no georeferencing – "
                "assigned synthetic EPSG:3857 CRS for comparison"
            )
            # Monkey-patch the dataset so downstream helpers work
            first_dataset._crs = _synth_crs_1
            first_dataset._transform = _synth_transform_1

        if not _has_georeferencing(second_dataset):
            _synth_crs_2, _synth_transform_2 = _synthesize_georeferencing(
                second_dataset
            )
            tracer.append_log(
                "after image has no georeferencing – "
                "assigned synthetic EPSG:3857 CRS for comparison"
            )
            second_dataset._crs = _synth_crs_2
            second_dataset._transform = _synth_transform_2

        first_modality = _detect_modality(
            first_dataset
        )

        second_modality = _detect_modality(
            second_dataset
        )

        tracer.append_log(
            "step 4: detected modalities "
            f"{first_modality} / {second_modality}"
        )

        # A temporal change pair must be same-modality.
        # Optical + SAR belongs to the cross-modal specialist path.
        if first_modality != second_modality:
            raise ValueError(
                "The two inputs appear to be different modalities "
                f"({first_modality} and {second_modality}). "
                "Use the cross-modal optical-SAR analysis path instead "
                "of the bi-temporal change detector."
            )

        overlap_fraction = _intersection_fraction(
            first_dataset.bounds,
            second_dataset.bounds,
        )

        tracer.append_log(
            "step 5: geographic overlap fraction "
            f"{overlap_fraction:.3f}"
        )

        if overlap_fraction <= 0:
            raise ValueError(
                "The before and after images do not overlap geographically."
            )

        if overlap_fraction < 0.50:
            tracer.append_log(
                "warning: overlap is below 50%; "
                "analysis will use the common intersection only"
            )

        # ---------------------------------------------------------------------
        # Common analysis grid
        # ---------------------------------------------------------------------

        grid = _make_common_grid(
            first_dataset,
            second_dataset,
        )

        tracer.append_log(
            "step 6: created common projected analysis grid "
            f"{grid['width']}x{grid['height']} "
            f"at {grid['resolution_x']:.3f} x "
            f"{grid['resolution_y']:.3f} map units"
        )

        # ---------------------------------------------------------------------
        # Select usable bands
        # ---------------------------------------------------------------------

        band_count = min(
            first_dataset.count,
            second_dataset.count,
        )

        if band_count <= 0:
            raise ValueError(
                "The rasters contain no usable bands."
            )

        indexes = list(
            range(
                1,
                band_count + 1,
            )
        )

        # ---------------------------------------------------------------------
        # Reproject both images to the same grid
        # ---------------------------------------------------------------------

        first_data, first_valid = (
            _reproject_to_grid(
                first_dataset,
                indexes,
                grid,
            )
        )

        second_data, second_valid = (
            _reproject_to_grid(
                second_dataset,
                indexes,
                grid,
            )
        )

        finite_pair = (
            first_valid
            & second_valid
            & np.all(np.isfinite(first_data), axis=0)
            & np.all(np.isfinite(second_data), axis=0)
        )

        if np.any(finite_pair):
           first_debug = first_data[:, finite_pair]
           second_debug = second_data[:, finite_pair]

           temporal_abs_difference = np.abs(
               second_debug - first_debug
            )

           tracer.append_log(
             "temporal array diagnostics: "
             f"mean_abs_diff={float(np.mean(temporal_abs_difference)):.8f}, "
             f"max_abs_diff={float(np.max(temporal_abs_difference)):.8f}, "
             f"nonzero_fraction={float(np.mean(temporal_abs_difference > 1e-8)):.6f}"
              )
        else:
             tracer.append_log(
            "temporal array diagnostics: NO FINITE COMMON PIXELS"
            )

        valid = (
            first_valid
            & second_valid
        )

        valid_fraction = (
            float(
                np.count_nonzero(valid)
            )
            /
            float(valid.size)
        )

        tracer.append_log(
            "step 7: valid common-pixel fraction "
            f"{valid_fraction:.3f}"
        )

        if valid_fraction < 0.10:
            raise ValueError(
                "Less than 10% of the common analysis grid contains "
                "valid pixels."
            )

        # ---------------------------------------------------------------------
        # Residual registration diagnostic
        # ---------------------------------------------------------------------

        shift_x, shift_y = (
            _estimate_translation(
                first_data,
                second_data,
                valid,
            )
        )

        shift_magnitude = math.sqrt(
            shift_x ** 2
            + shift_y ** 2
        )

        tracer.append_log(
            "step 8: residual registration estimate "
            f"{shift_x:.2f}px x, "
            f"{shift_y:.2f}px y "
            f"(magnitude {shift_magnitude:.2f}px)"
        )

        registration_warning = None

        if (
            shift_magnitude
            > MAX_REGISTRATION_SHIFT_PIXELS
        ):
            registration_warning = (
                "Large residual registration shift detected "
                f"({shift_magnitude:.2f} px). "
                "Change results may be unreliable."
            )

            tracer.append_log(
                "warning: "
                + registration_warning
            )

        elif shift_magnitude > 1.5:

            tracer.append_log(
                "warning: moderate residual registration shift detected"
            )

        # ---------------------------------------------------------------------
        # Change method
        # ---------------------------------------------------------------------

        feature = _query_feature(
            query
        )

        tracer.append_log(
            f"step 9: query feature interpreted as '{feature}'"
        )

        query_magnitude, query_method = (
            _query_specific_magnitude(
                feature,
                first_data,
                second_data,
                first_dataset,
                second_dataset,
                valid,
            )
        )

        if query_magnitude is not None:

            magnitude = query_magnitude

            tracer.append_log(
                "step 10: using query-specific "
                f"{query_method}"
            )

            detector_name = (
                query_method
            )

        elif first_modality == "sar":

            magnitude = _sar_log_ratio(
                first_data,
                second_data,
                valid,
            )

            detector_name = (
                "sar_log_ratio"
            )

            tracer.append_log(
                "step 10: using SAR log-ratio change magnitude"
            )

        else:

            # Optical pair: prefer IR-MAD.
            #
            # CVA measures raw radiometric difference, so a change in sun
            # angle, atmosphere or sensor calibration between the two dates
            # registers as change across the entire scene. IR-MAD is built on
            # canonical correlation, which is invariant to any affine
            # radiometric transform of either image, so only departures from
            # the shared linear relationship survive — which is what change
            # actually is. It also yields a chi-squared statistic with a
            # calibrated no-change probability rather than an arbitrary
            # magnitude scale.
            magnitude = None

            if irmad_module is not None:
                try:
                    irmad_result = irmad_module.irmad(
                        first_data,
                        second_data,
                        valid,
                    )
                    magnitude = irmad_result.chi_squared
                    detector_name = "optical_irmad"

                    tracer.append_log(
                        "step 10: using IR-MAD change magnitude "
                        f"(chi-squared, {first_data.shape[0]} bands, "
                        f"{irmad_result.iterations} iterations, "
                        f"converged={irmad_result.converged})"
                    )
                    tracer.append_log(
                        "step 10.1: canonical correlations "
                        f"{np.round(irmad_result.canonical_correlations, 4).tolist()}, "
                        f"calibration scale {irmad_result.calibration_scale:.3f}"
                    )
                except (np.linalg.LinAlgError, ValueError) as exc:
                    tracer.append_log(
                        f"step 10: IR-MAD unavailable for this pair ({exc}); "
                        "falling back to CVA"
                    )
            else:
                tracer.append_log(
                    "step 10: IR-MAD module not loaded "
                    f"({_IRMAD_IMPORT_ERROR}); falling back to CVA"
                )

            if magnitude is None:

                magnitude = _cva_change(
                   first_data,
                   second_data,
                   valid,
                )

                detector_name = (
                   "optical_cva"
                )

                tracer.append_log(
                  "step 10: using optical multiband CVA change magnitude"
                )

        # ---------------------------------------------------------------------
        # Optional ChangeFormer candidate
        # ---------------------------------------------------------------------
        #
        # ChangeFormer is NOT treated as the universal detector.
        #
        # It is used only when:
        #   1. the pair is optical,
        #   2. the query is general change,
        #   3. at least three bands are available.
        #
        # Feature-specific queries (water/built-up/vegetation) stay on their
        # corresponding spectral/index path. SAR stays on the SAR path.
        #
        # ChangeFormer supplies a spatial candidate mask. Deterministic
        # magnitude analysis remains responsible for measurement/verification.

        changeformer_mask = None
        changeformer_status = (
            "CHANGEFORMER_SKIPPED"
        )
        changeformer_candidate_fraction = 0.0

        if (
            first_modality == "optical"
            and feature == "general"
            and band_count >= 3
        ):
            (
                changeformer_mask,
                changeformer_status,
            ) = _try_changeformer(
                first_data,
                second_data,
                valid,
            )
        elif first_modality == "sar":
            changeformer_status = (
                "CHANGEFORMER_SKIPPED: SAR pair uses SAR-specific detector"
            )
        elif feature != "general":
            changeformer_status = (
                "CHANGEFORMER_SKIPPED: query-specific spectral analysis "
                "takes priority"
            )
        else:
            changeformer_status = (
                "CHANGEFORMER_SKIPPED: fewer than three usable bands"
            )

        tracer.append_log(
            "step 11: ChangeFormer status: "
            + changeformer_status
        )

        # ---------------------------------------------------------------------
        # Candidate selection + deterministic verification
        # ---------------------------------------------------------------------

        changeformer_candidate_accepted = False

        if changeformer_mask is not None:
            candidate_mask = (
                changeformer_mask.astype(bool)
                & valid
            )

            candidate_pixels = int(
                np.count_nonzero(candidate_mask)
            )

            valid_pixels_for_candidate = int(
                np.count_nonzero(valid)
            )

            changeformer_candidate_fraction = (
                candidate_pixels
                / max(
                    1,
                    valid_pixels_for_candidate,
                )
            )

            if (
                candidate_pixels > 0
                and changeformer_candidate_fraction
                <= MAX_CHANGEFORMER_CANDIDATE_FRACTION
            ):
                changed = candidate_mask
                changeformer_candidate_accepted = True

                finite_candidate_magnitude = magnitude[
                    candidate_mask
                    & np.isfinite(magnitude)
                ]

                if finite_candidate_magnitude.size > 0:
                    threshold = float(
                        np.nanmedian(
                            finite_candidate_magnitude
                        )
                    )
                else:
                    threshold = float("nan")

                threshold_method = (
                    "changeformer_candidate"
                )

                detector_name = (
                    "changeformer_candidate_with_cva_measurement"
                )

                tracer.append_log(
                    "step 12: accepted ChangeFormer candidate mask; "
                    f"candidate_fraction="
                    f"{changeformer_candidate_fraction:.4f}"
                )

            elif (
                candidate_pixels > 0
                and changeformer_candidate_fraction
                > MAX_CHANGEFORMER_CANDIDATE_FRACTION
            ):
                tracer.append_log(
                    "warning: ChangeFormer candidate covers "
                    f"{changeformer_candidate_fraction * 100:.2f}% "
                    "of valid pixels; treating it as suspicious and "
                    "falling back to deterministic thresholding"
                )

        if not changeformer_candidate_accepted:
            finite_magnitude = magnitude[
                valid
                & np.isfinite(magnitude)
            ]

            if finite_magnitude.size == 0:
                changed = np.zeros_like(
                    valid,
                    dtype=bool,
                )

                threshold = float("nan")

                threshold_method = (
                    "no_reliable_change_signal"
                )

            else:
                magnitude_min = float(
                    np.min(finite_magnitude)
                )

                magnitude_max = float(
                    np.max(finite_magnitude)
                )

                magnitude_range = (
                    magnitude_max
                    - magnitude_min
                )

                magnitude_tolerance = max(
                    CHANGE_SIGNAL_EPSILON,
                    abs(magnitude_max) * 1e-6,
                )

                if (
                   not np.isfinite(magnitude_min)
                   or not np.isfinite(magnitude_max)
                ):
                   changed = np.zeros_like(
                   valid,
                   dtype=bool,
                   )

                   threshold = float("nan")

                   threshold_method = (
                     "no_reliable_change_signal"
                    )

                   tracer.append_log(
                    "warning: change magnitude contains no usable finite range"
                   )

                else:
                    finite_debug = magnitude[
                      valid & np.isfinite(magnitude)
                    ]

                    if finite_debug.size > 0:
                       tracer.append_log(
                         "change magnitude diagnostics: "
                          f"min={float(np.min(finite_debug)):.6f}, "
                          f"max={float(np.max(finite_debug)):.6f}, "
                          f"mean={float(np.mean(finite_debug)):.6f}, "
                          f"median={float(np.median(finite_debug)):.6f}, "
                          f"p95={float(np.percentile(finite_debug, 95)):.6f}, "
                          f"p99={float(np.percentile(finite_debug, 99)):.6f}"
                        )
                    else:
                       tracer.append_log(
                          "change magnitude diagnostics: NO FINITE VALUES"
                        )
                    (
                      changed,
                      threshold,
                      threshold_method,
                    ) = _threshold_change(
                      magnitude,
                      valid,
                    )

            tracer.append_log(
                "step 12: deterministic thresholding method "
                f"{threshold_method}, "
                f"threshold={threshold:.6f}"
            )

        # ---------------------------------------------------------------------
        # Clean mask
        # ---------------------------------------------------------------------

        changed = _clean_mask(
            changed
        )

        changed &= valid

        changed_pixels = int(
            np.count_nonzero(
                changed
            )
        )

        valid_pixels = int(
            np.count_nonzero(
                valid
            )
        )

        change_fraction = (
            changed_pixels
            /
            max(
                1,
                valid_pixels,
            )
        )

        tracer.append_log(
            "step 13: cleaned change mask; "
            f"{changed_pixels} changed pixels "
            f"({change_fraction * 100:.2f}% of valid area)"
        )

        # ---------------------------------------------------------------------
        # Area calculation
        # ---------------------------------------------------------------------

        pixel_area_m2 = (
            grid["resolution_x"]
            * grid["resolution_y"]
        )

        changed_area_m2 = (
            changed_pixels
            * pixel_area_m2
        )

        changed_area_ha = (
            changed_area_m2
            / 10_000.0
        )

        valid_area_m2 = (
            valid_pixels
            * pixel_area_m2
        )

        # ---------------------------------------------------------------------
        # Polygonize
        # ---------------------------------------------------------------------

        polygons = _polygonize(
            changed,
            grid["transform"],
            grid["crs"],
        )

        tracer.append_log(
            "step 14: polygonized "
            f"{len(polygons)} changed region(s)"
        )

        # ---------------------------------------------------------------------
        # Output files
        # ---------------------------------------------------------------------

        analysis_id = (
            f"change_"
            f"{first_file.stem[:20]}_"
            f"{second_file.stem[:20]}_"
            f"{abs(hash(query)) % 1_000_000}"
        )

        output_folder = (
            OUTPUT_DIR
            / analysis_id
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        mask_path = (
            output_folder
            / "change_mask.tif"
        )

        magnitude_path = (
            output_folder
            / "change_magnitude.tif"
        )

        geojson_path = (
            output_folder
            / "changed_regions.geojson"
        )

        metadata_path = (
            output_folder
            / "analysis.json"
        )

        _save_mask_raster(
            changed,
            grid["transform"],
            grid["crs"],
            mask_path,
        )

        # Browser-drawable copy of the same mask.
        mask_overlay_path = (
            output_folder
            / "change_mask_web.png"
        )

        mask_overlay = _save_mask_web_overlay(
            changed,
            valid,
            grid["transform"],
            grid["crs"],
            mask_overlay_path,
        )

        _save_magnitude_raster(
            magnitude,
            grid["transform"],
            grid["crs"],
            magnitude_path,
        )

        geojson_data = _save_geojson(
            polygons,
            grid["crs"],
            geojson_path,
        )

        # ---------------------------------------------------------------------
        # Evidence summary
        # ---------------------------------------------------------------------

        magnitude_values = magnitude[
            valid
            & np.isfinite(magnitude)
        ]

        changed_magnitude_values = magnitude[
            changed
            & np.isfinite(magnitude)
        ]

        mean_change = (
            float(
                np.mean(
                    magnitude_values
                )
            )
            if magnitude_values.size
            else None
        )

        mean_changed_change = (
            float(
                np.mean(
                    changed_magnitude_values
                )
            )
            if changed_magnitude_values.size
            else None
        )

        max_change = (
            float(
                np.max(
                    magnitude_values
                )
            )
            if magnitude_values.size
            else None
        )

        min_change = (
            float(
                np.min(
                    magnitude_values
                )
            )
            if magnitude_values.size
            else None
        )

        median_change = (
            float(
                np.median(
                    magnitude_values
                )
            )
            if magnitude_values.size
            else None
        )

        p95_change = (
            float(
                np.percentile(
                    magnitude_values,
                    95,
                )
            )
            if magnitude_values.size
            else None
        )

        p99_change = (
            float(
                np.percentile(
                    magnitude_values,
                    99,
                )
            )
            if magnitude_values.size
            else None
        )

        change_signal_reliable = (
            threshold_method
            != "no_reliable_change_signal"
        )

        metadata = {
            "analysis_type": "bi_temporal_change_detection",
            "query": query,
            "feature": feature,
            "detector": detector_name,
            "before_image": first_file.name,
            "after_image": second_file.name,
            "before_modality": first_modality,
            "after_modality": second_modality,
            "before_crs": str(first_dataset.crs),
            "after_crs": str(second_dataset.crs),
            "analysis_crs": str(
                grid["crs"]
            ),
            "common_grid": {
                "width": grid["width"],
                "height": grid["height"],
                "resolution_x": grid[
                    "resolution_x"
                ],
                "resolution_y": grid[
                    "resolution_y"
                ],
            },
            "overlap_fraction": overlap_fraction,
            "valid_pixel_fraction": valid_fraction,
            "registration_shift_pixels": {
                "x": shift_x,
                "y": shift_y,
                "magnitude": shift_magnitude,
            },
            "registration_warning": registration_warning,
            "threshold": threshold,
            "threshold_method": threshold_method,
            "valid_pixels": valid_pixels,
            "changed_pixels": changed_pixels,
            "change_fraction": change_fraction,
            "pixel_area_m2": pixel_area_m2,
            "valid_area_m2": valid_area_m2,
            "changed_area_m2": changed_area_m2,
            "changed_area_hectares": changed_area_ha,
            "changed_regions": len(
                polygons
            ),
            "mean_change_magnitude": mean_change,
            "mean_changed_magnitude": mean_changed_change,
            "max_change_magnitude": max_change,
            "min_change_magnitude": min_change,
            "median_change_magnitude": median_change,
            "p95_change_magnitude": p95_change,
            "p99_change_magnitude": p99_change,
            "change_signal_reliable": change_signal_reliable,
            "changeformer_candidate_fraction": (
                changeformer_candidate_fraction
            ),
            "changeformer_candidate_accepted": (
                changeformer_candidate_accepted
            ),
            "changeformer_status": changeformer_status,
            "outputs": {
                "change_mask": str(
                    mask_path
                ),
                "change_magnitude": str(
                    magnitude_path
                ),
                "changed_regions_geojson": str(
                    geojson_path
                ),
            },
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        # ---------------------------------------------------------------------
        # Human-readable answer
        # ---------------------------------------------------------------------

        if changed_pixels == 0:

            if threshold_method == "no_reliable_change_signal":
                answer = (
                    "No reliable spatial change signal was detected "
                    "within the valid overlapping area. The available "
                    "change-magnitude distribution was too weak or "
                    "degenerate to produce a defensible change mask."
                )
            else:
                answer = (
                    "No significant spatial change was detected "
                    "within the valid overlapping area."
                )

        else:

            direction_text = ""

            if feature == "water":
                direction_text = (
                    " The analysis measures observable water-extent "
                    "change only; it does not estimate water depth, "
                    "water level, or water volume."
                )

            elif feature == "built_up":
                direction_text = (
                    " The result measures spatial change associated "
                    "with the built-up signal; it is not a direct "
                    "building-count measurement."
                )

            elif feature == "vegetation":
                direction_text = (
                    " The result measures change in vegetation "
                    "spectral signal where the required bands were available."
                )

            answer = (
                f"Detected approximately "
                f"{changed_area_ha:.2f} hectares "
                f"({changed_area_m2:.0f} m²) of changed area "
                f"within the valid overlapping region, representing "
                f"{change_fraction * 100:.2f}% of valid pixels."
                f"{direction_text}"
            )

        if overlap_fraction < 1.0:
            answer += (
                f" The images overlap over approximately "
                f"{overlap_fraction * 100:.1f}% of the smaller "
                f"scene footprint, and analysis was restricted "
                f"to the common area."
            )

        if registration_warning:
            answer += (
                " Warning: residual registration error is high, "
                "so the spatial change result should be treated cautiously."
            )

        # ---------------------------------------------------------------------
        # Evidence paths returned to frontend
        # ---------------------------------------------------------------------

        evidence: List[Any] = [
            geojson_data,
        ]

        if mask_overlay is not None:
            # Origin-relative so it resolves through whatever host is serving
            # the API. outputs/ is mounted at /static/outputs by main.py.
            relative = mask_overlay_path.relative_to(OUTPUT_DIR.parent).as_posix()
            evidence.append(
                {
                    "type": "ImageOverlay",
                    "label": "Detected change",
                    "url": f"/static/outputs/{relative}",
                    "wgs84_bounds": mask_overlay["wgs84_bounds"],
                    "opacity": 0.7,
                }
            )
            tracer.append_log(
                "step 15.1: wrote Web-Mercator change-mask overlay "
                f"({mask_overlay['width']}x{mask_overlay['height']} px)"
            )

        evidence.extend(
            [
                str(mask_path),
                str(magnitude_path),
                str(metadata_path),
            ]
        )

        tracer.append_log(
            "step 15: saved spatial evidence and analysis metadata"
        )

        return (
            answer,
            evidence,
        )


# =============================================================================
# Public service API
# =============================================================================

async def run_inference(
    query: str,
    file_paths: List[str],
    tracer: Any,
) -> Tuple[str, List[Any]]:

    """
    Public Change Detective entry point used by core.agent.

    Expected temporal input:
        [before.tif, after.tif]
    """

    if len(file_paths) != 2:
        return (
            "Bi-temporal change detection requires exactly two "
            "spatially corresponding images: a before image and an after image.",
            [],
        )

    tracer.append_log(
        "Change Detective received exactly two temporal inputs"
    )

    try:

        result = await asyncio.to_thread(
            _analyse_pair,
            query,
            file_paths[0],
            file_paths[1],
            tracer,
        )

        tracer.append_log(
            "Change Detective completed successfully"
        )

        return result

    except Exception as exc:

        tracer.append_log(
            "Change Detective failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return (
            "Change analysis could not be completed. "
            f"Reason: {exc}",
            [],
        )