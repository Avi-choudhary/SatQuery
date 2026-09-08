import json
from pathlib import Path
from config import METADATA_DIR
from src.io_utils import load_raster


class PatchIndex:
    def __init__(self, metadata_dir=METADATA_DIR):
        """Load all the metadata JSON files into memory once."""
        self.metadata_dir = Path(metadata_dir)
        self.records = []
        self.reload()

    def reload(self):
        """Reload records from the metadata directory."""
        self.records = []
        if self.metadata_dir.exists():
            for json_path in sorted(self.metadata_dir.glob("*.json")):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        self.records.append(json.load(f))
                except Exception as e:
                    print(f"Warning: Failed to load {json_path}: {e}")
        return len(self.records)

    def query(self, region=None, sensor=None, date=None, patch_id=None):
        results = self.records
        if region:
            results = [r for r in results if r.get("region") == region]
        if sensor:
            results = [r for r in results if r.get("sensor") == sensor]
        if date:
            results = [r for r in results if r.get("date") == date]
        if patch_id is not None:
            results = [r for r in results if r.get("patch_id") == patch_id]
        return results

    def query_bbox(self, min_lon, min_lat, max_lon, max_lat, region=None, sensor=None, date=None):
        """Find patches that spatially intersect the given bounding box."""
        candidates = self.query(region=region, sensor=sensor, date=date)
        matched = []
        for r in candidates:
            box = r.get("bounding_box") or r.get("wgs84_bounds")
            if not box or len(box) != 4:
                continue
            r_min_lon, r_min_lat, r_max_lon, r_max_lat = box
            # Check overlap: box A overlaps box B if not disjoint
            disjoint = (
                r_max_lon < min_lon or
                r_min_lon > max_lon or
                r_max_lat < min_lat or
                r_min_lat > max_lat
            )
            if not disjoint:
                matched.append(r)
        return matched

    def query_point(self, lat, lon, region=None, sensor=None, date=None):
        """Find patches whose bounding box contains the specified (lat, lon) point."""
        candidates = self.query(region=region, sensor=sensor, date=date)
        matched = []
        for r in candidates:
            box = r.get("bounding_box") or r.get("wgs84_bounds")
            if not box or len(box) != 4:
                continue
            r_min_lon, r_min_lat, r_max_lon, r_max_lat = box
            if r_min_lon <= lon <= r_max_lon and r_min_lat <= lat <= r_max_lat:
                matched.append(r)
        return matched

    def to_geojson(self, records=None):
        """Export records as a standard GeoJSON FeatureCollection ready for Mapbox/Leaflet."""
        if records is None:
            records = self.records

        features = []
        for r in records:
            geom = r.get("geojson")
            if not geom:
                box = r.get("bounding_box") or r.get("wgs84_bounds")
                if box and len(box) == 4:
                    geom = {
                        "type": "Polygon",
                        "coordinates": [[
                            [box[0], box[1]],
                            [box[2], box[1]],
                            [box[2], box[3]],
                            [box[0], box[3]],
                            [box[0], box[1]]
                        ]]
                    }
            if not geom:
                continue

            properties = {
                "patch_id": r.get("patch_id"),
                "name": r.get("name"),
                "region": r.get("region"),
                "date": r.get("date"),
                "sensor": r.get("sensor"),
                "center": r.get("center"),
                "shape": r.get("shape"),
                "png_url": r.get("png_url"),
                "tif_path": r.get("tif_path"),
            }
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": properties
            })

        return {
            "type": "FeatureCollection",
            "features": features
        }

    def get_patch_array(self, record):
        array, _ = load_raster(record["tif_path"])
        return array

    def get_pair(self, region, date):
        """Retrieve matching optical (S2) and SAR (S1) arrays for a given region and date."""
        optical_records = self.query(region=region, sensor="S2", date=date)
        sar_records = self.query(region=region, sensor="S1", date=date)
        if not optical_records or not sar_records:
            raise ValueError(f"No pair found for region='{region}', date='{date}'.")
        return {
            "optical": self.get_patch_array(optical_records[0]),
            "sar": self.get_patch_array(sar_records[0]),
            "metadata": {"region": region, "date": date},
        }

    def get_timeseries(self, region, sensor="S2"):
        """Get all dates for one region, sorted for change detection."""
        records = sorted(self.query(region=region, sensor=sensor), key=lambda r: r["date"])
        return [{"array": self.get_patch_array(r), "metadata": r} for r in records]