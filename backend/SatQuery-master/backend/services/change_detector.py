from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage


def robust_scale_unit(
    image: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(image, dtype=np.float32)
    values = image[valid_mask & np.isfinite(image)]
    if values.size < 10:
        return output
    low, high = np.percentile(values, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return output
    output[valid_mask] = np.clip((image[valid_mask] - low) / (high - low), 0.0, 1.0)
    return output


def prepare_unit_imagery(
    imagery: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    prepared = np.zeros_like(imagery, dtype=np.float32)
    for band_idx in range(imagery.shape[0]):
        prepared[band_idx] = robust_scale_unit(imagery[band_idx], valid_mask)
    return prepared


def relative_radiometric_normalize(
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    normalized_after = np.copy(after)
    bands = before.shape[0]
    diagnostics: Dict[str, Any] = {
        "method": "none",
        "stable_pixels": 0,
        "band_adjustments": [],
    }

    valid_coords = valid_mask & np.all(np.isfinite(before), axis=0) & np.all(np.isfinite(after), axis=0)
    valid_count = int(np.count_nonzero(valid_coords))
    if valid_count < 32:
        diagnostics["method"] = "insufficient_valid_pixels"
        return normalized_after, "insufficient_valid_pixels", diagnostics

    mean_before = np.mean(before, axis=0)
    mean_after = np.mean(after, axis=0)
    brightness_difference = np.abs(mean_after - mean_before)
    valid_differences = brightness_difference[valid_coords]

    p15_threshold = float(np.percentile(valid_differences, 15))
    pif_mask = valid_coords & (brightness_difference <= p15_threshold)
    pif_count = int(np.count_nonzero(pif_mask))
    diagnostics["stable_pixels"] = pif_count

    if pif_count >= 64:
        method = "pif_linear"
        for band_idx in range(bands):
            x_vals = after[band_idx, pif_mask]
            y_vals = before[band_idx, pif_mask]
            var_x = float(np.var(x_vals))
            gain = 1.0
            offset = 0.0
            fit_type = "pif"

            if var_x > 1e-7:
                covariance = float(np.cov(x_vals, y_vals)[0, 1])
                candidate_gain = covariance / var_x
                candidate_offset = float(np.mean(y_vals) - candidate_gain * np.mean(x_vals))
                if 0.3 <= candidate_gain <= 3.0:
                    gain = candidate_gain
                    offset = candidate_offset

            if gain == 1.0 and offset == 0.0:
                std_x = float(np.std(x_vals))
                std_y = float(np.std(y_vals))
                gain = std_y / max(std_x, 1e-6)
                gain = float(np.clip(gain, 0.3, 3.0))
                offset = float(np.mean(y_vals) - gain * np.mean(x_vals))
                fit_type = "pif_moments"

            adjusted_band = gain * after[band_idx, valid_mask] + offset
            normalized_after[band_idx, valid_mask] = np.clip(adjusted_band, 0.0, 1.0)
            diagnostics["band_adjustments"].append(
                {"band": band_idx + 1, "gain": gain, "offset": offset, "fit_type": fit_type}
            )
    else:
        method = "histogram_moments_fallback"
        for band_idx in range(bands):
            x_vals = after[band_idx, valid_coords]
            y_vals = before[band_idx, valid_coords]
            std_x = float(np.std(x_vals))
            std_y = float(np.std(y_vals))
            gain = float(np.clip(std_y / max(std_x, 1e-6), 0.3, 3.0))
            offset = float(np.mean(y_vals) - gain * np.mean(x_vals))
            adjusted_band = gain * after[band_idx, valid_mask] + offset
            normalized_after[band_idx, valid_mask] = np.clip(adjusted_band, 0.0, 1.0)
            diagnostics["band_adjustments"].append(
                {"band": band_idx + 1, "gain": gain, "offset": offset, "fit_type": "moments_fallback"}
            )

    diagnostics["method"] = method
    return normalized_after, method, diagnostics


def optical_cva(
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if before.shape != after.shape:
        raise ValueError("Before and after arrays must have identical shapes.")
    if before.ndim != 3:
        raise ValueError("Expected imagery with shape (bands, height, width).")

    difference = after - before
    magnitude = np.sqrt(np.sum(difference ** 2, axis=0))
    direction = np.zeros_like(difference, dtype=np.float32)

    safe_mag = magnitude + 1e-6
    for band_idx in range(before.shape[0]):
        direction[band_idx] = difference[band_idx] / safe_mag

    magnitude[~valid_mask] = np.nan
    direction[:, ~valid_mask] = np.nan
    return magnitude.astype(np.float32), direction.astype(np.float32)


def sar_log_ratio(
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if before.shape != after.shape:
        raise ValueError("Before and after arrays must have identical shapes.")
    if before.ndim != 3:
        raise ValueError("Expected imagery with shape (bands, height, width).")

    before_safe = np.maximum(np.abs(before), 1e-6)
    after_safe = np.maximum(np.abs(after), 1e-6)

    before_log = np.log(before_safe)
    after_log = np.log(after_safe)

    difference = after_log - before_log
    magnitude = np.sqrt(np.sum(difference ** 2, axis=0))
    direction = np.zeros_like(difference, dtype=np.float32)

    safe_mag = magnitude + 1e-6
    for band_idx in range(before.shape[0]):
        direction[band_idx] = difference[band_idx] / safe_mag

    magnitude[~valid_mask] = np.nan
    direction[:, ~valid_mask] = np.nan
    return magnitude.astype(np.float32), direction.astype(np.float32)


def otsu_threshold(
    values: np.ndarray,
) -> Optional[float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 32:
        return None

    low, high = np.percentile(finite_values, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None

    clipped = np.clip(finite_values, low, high)
    histogram, edges = np.histogram(clipped, bins=256, range=(low, high))
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return None

    probabilities = histogram / total
    centres = (edges[:-1] + edges[1:]) / 2.0

    omega = np.cumsum(probabilities)
    mu = np.cumsum(probabilities * centres)
    total_mean = mu[-1]

    denominator = omega * (1.0 - omega)
    numerator = (total_mean * omega - mu) ** 2

    between_class_variance = np.zeros_like(numerator)
    safe = denominator > 1e-12
    between_class_variance[safe] = numerator[safe] / denominator[safe]
    if not np.any(safe):
        return None

    threshold_idx = int(np.argmax(between_class_variance))
    threshold = float(centres[threshold_idx])
    dynamic_range = high - low

    if threshold <= low + 0.01 * dynamic_range or threshold >= high - 0.01 * dynamic_range:
        return None

    return threshold


def robust_threshold(
    magnitude: np.ndarray,
    valid_mask: np.ndarray,
    sanity_bound: float = 0.35,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, float, str, Dict[str, Any]]:
    values = magnitude[valid_mask & np.isfinite(magnitude)]
    empty_mask = np.zeros_like(valid_mask, dtype=bool)

    diag: Dict[str, Any] = {
        "valid_pixel_count": int(np.count_nonzero(valid_mask)),
        "min_magnitude": 0.0,
        "max_magnitude": 0.0,
        "median_magnitude": 0.0,
        "mad": 0.0,
        "sanity_triggered": False,
    }

    if values.size < 32:
        return empty_mask, float("nan"), "no_reliable_change_signal", diag

    min_val = float(np.min(values))
    max_val = float(np.max(values))
    diag["min_magnitude"] = min_val
    diag["max_magnitude"] = max_val

    dynamic_range = max_val - min_val
    tolerance = max(epsilon, abs(max_val) * 1e-5)
    if dynamic_range <= tolerance or max_val <= epsilon:
        return empty_mask, float("nan"), "no_reliable_change_signal", diag

    median_val = float(np.median(values))
    mad_val = float(1.4826 * np.median(np.abs(values - median_val)))
    diag["median_magnitude"] = median_val
    diag["mad"] = mad_val

    otsu_cand = otsu_threshold(values)
    selected_threshold = float("nan")
    selected_method = "mad_robust"

    mad_cand = median_val + 2.5 * mad_val

    if otsu_cand is not None and otsu_cand > median_val:
        p_otsu = float(np.mean(values >= otsu_cand))
        if 0.001 <= p_otsu <= sanity_bound:
            selected_threshold = otsu_cand
            selected_method = "otsu"

    if not np.isfinite(selected_threshold):
        if mad_cand > min_val and mad_cand < max_val:
            selected_threshold = mad_cand
            selected_method = "mad_robust"
        else:
            selected_threshold = float(np.percentile(values, 95))
            selected_method = "percentile_95"

    mask = (magnitude >= selected_threshold) & valid_mask
    valid_count = max(1, int(np.count_nonzero(valid_mask)))
    changed_count = int(np.count_nonzero(mask))
    changed_fraction = changed_count / valid_count

    if changed_fraction > sanity_bound:
        diag["sanity_triggered"] = True
        stricter_threshold = median_val + 3.5 * mad_val
        stricter_fraction = float(np.mean(values >= stricter_threshold))
        if 0 < stricter_fraction <= sanity_bound and stricter_threshold < max_val:
            selected_threshold = stricter_threshold
            selected_method = "mad_stricter_sanity"
            mask = (magnitude >= selected_threshold) & valid_mask
        else:
            p99_cand = float(np.percentile(values, 99))
            p99_fraction = float(np.mean(values >= p99_cand))
            if p99_fraction <= sanity_bound and p99_cand > min_val + tolerance:
                selected_threshold = p99_cand
                selected_method = "percentile_99_sanity"
                mask = (magnitude >= selected_threshold) & valid_mask
            else:
                return empty_mask, float("nan"), "scene_sanity_exceeded", diag

    return mask.astype(bool), float(selected_threshold), selected_method, diag


def clean_change_mask(
    change_mask: np.ndarray,
    minimum_region_pixels: int = 9,
) -> np.ndarray:
    opened = ndimage.binary_opening(
        change_mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    closed = ndimage.binary_closing(
        opened,
        structure=np.ones((3, 3), dtype=bool),
    )

    labels, num_features = ndimage.label(
        closed,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if num_features == 0:
        return closed.astype(bool)

    sizes = np.bincount(labels.ravel())
    keep_indices = sizes >= minimum_region_pixels
    keep_indices[0] = False
    cleaned = keep_indices[labels]
    return cleaned.astype(bool)


def calculate_change_statistics(
    change_magnitude: np.ndarray,
    change_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    valid_values = change_magnitude[valid_mask & np.isfinite(change_magnitude)]
    changed_values = change_magnitude[change_mask & np.isfinite(change_magnitude)]

    valid_pixels = int(np.count_nonzero(valid_mask))
    changed_pixels = int(np.count_nonzero(change_mask))
    change_fraction = float(changed_pixels / max(1, valid_pixels))

    stats: Dict[str, Any] = {
        "valid_pixels": valid_pixels,
        "changed_pixels": changed_pixels,
        "change_fraction": change_fraction,
        "mean_change": float(np.mean(valid_values)) if valid_values.size else 0.0,
        "median_change": float(np.median(valid_values)) if valid_values.size else 0.0,
        "mean_changed_change": float(np.mean(changed_values)) if changed_values.size else 0.0,
        "max_change": float(np.max(valid_values)) if valid_values.size else 0.0,
        "min_change": float(np.min(valid_values)) if valid_values.size else 0.0,
        "p95_change": float(np.percentile(valid_values, 95)) if valid_values.size else 0.0,
        "p99_change": float(np.percentile(valid_values, 99)) if valid_values.size else 0.0,
    }
    return stats