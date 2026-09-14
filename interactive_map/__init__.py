"""
Interactive Map Module for SatQuery
Provides AOI bounding-box satellite imagery querying via STAC and windowed COG streaming.
"""

from .fetch_scene import fetch_scene

__all__ = ["fetch_scene"]
