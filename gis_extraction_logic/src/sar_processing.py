import numpy as np
from scipy.ndimage import uniform_filter


def linear_to_db(array, epsilon=1e-10):
    """Radar values start in a raw 'linear' scale that's hard to work
    with. Converting to decibels (dB) — like how we measure sound —
    makes the range much more manageable."""
    return 10.0 * np.log10(np.clip(array, epsilon, None))


def lee_filter(array, window_size=5):
    """This is the noise-reducer. In flat/uniform areas it smooths
    aggressively (since noise sticks out there). Near edges/structures
    it barely touches the pixel (so real detail survives)."""
    img_mean = uniform_filter(array, size=window_size)
    img_sqr_mean = uniform_filter(array**2, size=window_size)
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = np.var(array)
    weights = img_variance / (img_variance + overall_variance + 1e-10)
    return img_mean + weights * (array - img_mean)


def normalize_sar_band(array, db_min=-25.0, db_max=5.0):
    """Squash dB values into 0-1 using FIXED bounds (not per-image
    min/max) so the same real backscatter always maps to the same
    number, no matter which tile it's in."""
    clipped = np.clip(array, db_min, db_max)
    return (clipped - db_min) / (db_max - db_min)


def process_sar_tile(array, window_size=5):
    """Full recipe, run once per polarization band (VV, VH)."""
    db = linear_to_db(array)
    filtered = np.stack([lee_filter(db[i], window_size) for i in range(db.shape[0])])
    normalized = np.stack([normalize_sar_band(filtered[i]) for i in range(filtered.shape[0])])
    return normalized.astype(np.float32)