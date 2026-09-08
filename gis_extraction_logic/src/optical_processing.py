import numpy as np
from scipy.ndimage import zoom
from config import TARGET_RESOLUTION


def resample_band(array, src_res, target_res=TARGET_RESOLUTION):
    """If a band is blurrier (bigger res) than our target, stretch it
    to match — so all bands end up the same size and can be stacked."""
    if src_res == target_res:
        return array
    scale_factor = src_res / target_res
    new_shape = (int(array.shape[-2]*scale_factor), int(array.shape[-1]*scale_factor))
    zoom_factors = (new_shape[0]/array.shape[-2], new_shape[1]/array.shape[-1])
    return zoom(array, zoom_factors, order=1)   # order=1 = smooth interpolation


def apply_cloud_mask(array, cloud_mask, fill_value=np.nan):
    """Blank out pixels that are covered by cloud, using a cloud
    flag layer if your download includes one. Optional step."""
    masked = array.astype(float).copy()
    masked[..., cloud_mask] = fill_value
    return masked


def normalize_reflectance(array, scale_factor=10000.0):
    """Squash raw values (0-10000) down into a friendly 0-1 range."""
    normalized = array.astype(np.float32) / scale_factor
    return np.clip(normalized, 0.0, 1.0)


def process_optical_tile(array, cloud_mask=None):
    """The full recipe for one optical tile, in order."""
    if cloud_mask is not None:
        array = apply_cloud_mask(array, cloud_mask)
    return normalize_reflectance(array)