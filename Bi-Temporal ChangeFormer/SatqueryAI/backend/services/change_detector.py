from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _robust_scale(
    image: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Robustly scale each band to 0-1 using the 2nd and 98th percentiles.
    """

    output = np.zeros_like(
        image,
        dtype=np.float32,
    )

    values = image[
        valid_mask
        & np.isfinite(image)
    ]

    if values.size < 10:
        return output

    low, high = np.percentile(
        values,
        [2, 98],
    )

    if not np.isfinite(low) or not np.isfinite(high):
        return output

    if high <= low:
        output[
            valid_mask
        ] = 0.0
        return output

    output[valid_mask] = np.clip(
        (
            image[valid_mask] - low
        )
        / (
            high - low
        ),
        0.0,
        1.0,
    )

    return output


def optical_cva(
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Change Vector Analysis for optical imagery.

    Parameters
    ----------
    before:
        Array shaped (bands, height, width).

    after:
        Array shaped (bands, height, width).

    valid_mask:
        Common valid-pixel mask.

    Returns
    -------
    np.ndarray
        Continuous change magnitude raster.
    """

    if before.shape != after.shape:
        raise ValueError(
            "Before and after arrays must have identical shapes."
        )

    if before.ndim != 3:
        raise ValueError(
            "Expected imagery with shape "
            "(bands, height, width)."
        )

    scaled_before = np.zeros_like(
        before,
        dtype=np.float32,
    )

    scaled_after = np.zeros_like(
        after,
        dtype=np.float32,
    )

    for band in range(
        before.shape[0]
    ):
        scaled_before[band] = _robust_scale(
            before[band],
            valid_mask,
        )

        scaled_after[band] = _robust_scale(
            after[band],
            valid_mask,
        )

    difference = (
        scaled_after
        - scaled_before
    )

    magnitude = np.sqrt(
        np.sum(
            difference ** 2,
            axis=0,
        )
    )

    magnitude[
        ~valid_mask
    ] = np.nan

    return magnitude.astype(
        np.float32
    )


def sar_log_ratio(
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    SAR change magnitude using log-ratio differencing.

    This is intended for SAR intensity/backscatter data and should
    not be applied to optical imagery.
    """

    if before.shape != after.shape:
        raise ValueError(
            "Before and after arrays must have identical shapes."
        )

    if before.ndim != 3:
        raise ValueError(
            "Expected imagery with shape "
            "(bands, height, width)."
        )

    before_safe = np.maximum(
        np.abs(before),
        1e-6,
    )

    after_safe = np.maximum(
        np.abs(after),
        1e-6,
    )

    before_log = np.log(
        before_safe
    )

    after_log = np.log(
        after_safe
    )

    difference = (
        after_log
        - before_log
    )

    magnitude = np.sqrt(
        np.sum(
            difference ** 2,
            axis=0,
        )
    )

    magnitude[
        ~valid_mask
    ] = np.nan

    return magnitude.astype(
        np.float32
    )


def normalised_difference(
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Generic optical/feature-space change detector.

    Uses the same robust multiband representation as optical CVA.
    """

    return optical_cva(
        before,
        after,
        valid_mask,
    )


def otsu_threshold(
    change_magnitude: np.ndarray,
    valid_mask: np.ndarray,
) -> Optional[float]:
    """
    Calculate an Otsu threshold from a continuous change-magnitude raster.

    Returns None when the data distribution is unsuitable for a stable
    threshold.
    """

    values = change_magnitude[
        valid_mask
        & np.isfinite(change_magnitude)
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

    probability = (
        histogram / total
    )

    centres = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    cumulative_probability = np.cumsum(
        probability
    )

    cumulative_mean = np.cumsum(
        probability
        * centres
    )

    global_mean = cumulative_mean[-1]

    denominator = (
        cumulative_probability
        * (
            1.0
            - cumulative_probability
        )
    )

    numerator = (
        global_mean
        * cumulative_probability
        - cumulative_mean
    ) ** 2

    between_class_variance = np.zeros_like(
        numerator
    )

    valid = denominator > 1e-12

    between_class_variance[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    if not np.any(valid):
        return None

    threshold_index = int(
        np.argmax(
            between_class_variance
        )
    )

    threshold = float(
        centres[threshold_index]
    )

    dynamic_range = high - low

    # Reject thresholds that are effectively at the extremes.
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


def create_change_mask(
    change_magnitude: np.ndarray,
    valid_mask: np.ndarray,
    percentile_fallback: float = 95.0,
) -> Tuple[np.ndarray, float, str]:
    """
    Convert continuous change magnitude into a binary change mask.

    Otsu is preferred. A percentile fallback is used when the distribution
    does not support a reliable Otsu split.
    """

    threshold = otsu_threshold(
        change_magnitude,
        valid_mask,
    )

    method = "otsu"

    if threshold is None:

        values = change_magnitude[
            valid_mask
            & np.isfinite(change_magnitude)
        ]

        if values.size == 0:
            raise ValueError(
                "No valid pixels available for change thresholding."
            )

        threshold = float(
            np.percentile(
                values,
                percentile_fallback,
            )
        )

        method = (
            f"percentile_{percentile_fallback:g}"
        )

    mask = (
        change_magnitude
        >= threshold
    )

    mask &= valid_mask

    return (
        mask.astype(bool),
        float(threshold),
        method,
    )


def clean_change_mask(
    change_mask: np.ndarray,
    minimum_region_pixels: int = 9,
) -> np.ndarray:
    """
    Remove isolated noise and very small connected components.
    """

    try:
        from scipy import ndimage

    except ImportError:
        return change_mask.astype(bool)

    cleaned = ndimage.binary_opening(
        change_mask,
        structure=np.ones(
            (3, 3),
            dtype=bool,
        ),
    )

    cleaned = ndimage.binary_closing(
        cleaned,
        structure=np.ones(
            (3, 3),
            dtype=bool,
        ),
    )

    labels, number_of_regions = ndimage.label(
        cleaned,
        structure=np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
    )

    if number_of_regions == 0:
        return cleaned.astype(bool)

    region_sizes = np.bincount(
        labels.ravel()
    )

    keep = (
        region_sizes
        >= minimum_region_pixels
    )

    keep[0] = False

    cleaned = keep[
        labels
    ]

    return cleaned.astype(bool)


def calculate_change_statistics(
    change_magnitude: np.ndarray,
    change_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Produce evidence statistics for the detected change.
    """

    valid_values = change_magnitude[
        valid_mask
        & np.isfinite(change_magnitude)
    ]

    changed_values = change_magnitude[
        change_mask
        & np.isfinite(change_magnitude)
    ]

    valid_pixels = int(
        np.count_nonzero(
            valid_mask
        )
    )

    changed_pixels = int(
        np.count_nonzero(
            change_mask
        )
    )

    statistics = {
        "valid_pixels": float(
            valid_pixels
        ),
        "changed_pixels": float(
            changed_pixels
        ),
        "change_fraction": float(
            changed_pixels
            / max(
                1,
                valid_pixels,
            )
        ),
        "mean_change": float(
            np.mean(valid_values)
        )
        if valid_values.size
        else 0.0,
        "mean_changed_change": float(
            np.mean(changed_values)
        )
        if changed_values.size
        else 0.0,
        "max_change": float(
            np.max(valid_values)
        )
        if valid_values.size
        else 0.0,
    }

    return statistics