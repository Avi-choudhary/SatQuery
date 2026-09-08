import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from config import TARGET_CRS, TARGET_RESOLUTION


def build_target_grid(src_profile, target_crs=TARGET_CRS, target_resolution=TARGET_RESOLUTION):
    """Figure out what the 'shared grid' should look like, based on
    one image's profile (we use optical as the reference)."""
    transform, width, height = calculate_default_transform(
        src_profile["crs"], target_crs, src_profile["width"], src_profile["height"],
        *rasterio.transform.array_bounds(src_profile["height"], src_profile["width"], src_profile["transform"]),
        resolution=target_resolution,
    )
    target_profile = src_profile.copy()
    target_profile.update(crs=target_crs, transform=transform, width=width, height=height)
    return target_profile


def reproject_to_grid(array, src_profile, target_profile, resampling_method=Resampling.bilinear):
    """Actually warp one image onto the target grid, band by band."""
    band_count = array.shape[0]
    out_array = np.zeros((band_count, target_profile["height"], target_profile["width"]), dtype=array.dtype)
    for i in range(band_count):
        reproject(
            source=array[i], destination=out_array[i],
            src_transform=src_profile["transform"], src_crs=src_profile["crs"],
            dst_transform=target_profile["transform"], dst_crs=target_profile["crs"],
            resampling=resampling_method,
        )
    return out_array


def coregister_pair(optical_array, optical_profile, sar_array, sar_profile):
    """Warp BOTH optical and SAR onto the same shared grid (built from optical)."""
    target_profile = build_target_grid(optical_profile)
    aligned_optical = reproject_to_grid(optical_array, optical_profile, target_profile)
    aligned_sar = reproject_to_grid(sar_array, sar_profile, target_profile)
    return aligned_optical, aligned_sar, target_profile


def verify_alignment(profile_a, profile_b):
    """Sanity check: do these two images actually share the same grid now?"""
    return all([
        profile_a["crs"] == profile_b["crs"],
        profile_a["transform"] == profile_b["transform"],
        profile_a["width"] == profile_b["width"],
        profile_a["height"] == profile_b["height"],
    ])