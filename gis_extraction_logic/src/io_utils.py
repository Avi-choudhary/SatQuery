from pathlib import Path
from dataclasses import dataclass
import rasterio
import numpy as np


@dataclass
class RasterInfo:
    path: Path
    crs: str
    width: int
    height: int
    resolution: tuple
    bounds: tuple
    band_count: int
    dtype: str


def inspect_raster(path):
    """Peek at a file's metadata WITHOUT loading the huge pixel data.
    Always run this first on a new file — tells you its resolution/CRS
    so you know if it needs fixing before you process it."""
    path = Path(path)
    with rasterio.open(path) as src:
        info = RasterInfo(
            path=path, crs=str(src.crs), width=src.width, height=src.height,
            resolution=src.res, bounds=src.bounds,
            band_count=src.count, dtype=src.dtypes[0],
        )
    return info


def load_raster(path, bands=None):
    """Actually load the pixel values into a numpy array.
    Shape comes back as (bands, height, width) — e.g. a 4-band
    optical image at 256x256 is shape (4, 256, 256)."""
    path = Path(path)
    with rasterio.open(path) as src:
        array = src.read() if bands is None else src.read(bands)
        profile = src.profile.copy()   # metadata needed to save it back out later
    return array, profile


def save_raster(path, array, profile):
    """Write a numpy array back out as a proper georeferenced file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update(count=array.shape[0] if array.ndim == 3 else 1,
                    height=array.shape[-2], width=array.shape[-1], dtype=array.dtype)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array) if array.ndim == 3 else dst.write(array, 1)


def list_tiffs(directory):
    """Get all .tif files in a folder, sorted, so processing order is predictable."""
    directory = Path(directory)
    return sorted(list(directory.glob("*.tif")) + list(directory.glob("*.tiff")))