from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from config import MAX_PAIR_GAP_DAYS


@dataclass
class TileRecord:
    path: Path
    region: str
    date: datetime
    sensor: str


def parse_filename(path):
    """Expects files named like: delhi_20260112_S2.tif"""
    parts = path.stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"'{path.name}' doesn't match region_date_sensor.tif")
    region, date_str, sensor = parts
    return TileRecord(path=path, region=region, date=datetime.strptime(date_str, "%Y%m%d"), sensor=sensor)


def build_index(paths):
    records = []
    for p in paths:
        try:
            records.append(parse_filename(p))
        except ValueError as e:
            print(f"Skipping: {e}")
    return records


def find_cross_modal_pairs(records, max_gap_days=MAX_PAIR_GAP_DAYS):
    """Match an optical file with a SAR file — same region, close dates."""
    optical = [r for r in records if r.sensor == "S2"]
    sar = [r for r in records if r.sensor == "S1"]
    pairs = []
    for opt in optical:
        for s in sar:
            if opt.region == s.region and abs((opt.date - s.date).days) <= max_gap_days:
                pairs.append((opt, s))
    return pairs


def find_multitemporal_pairs(records, sensor="S2"):
    """Match two dates of the SAME sensor and region — for change detection."""
    by_region = {}
    for r in records:
        if r.sensor == sensor:
            by_region.setdefault(r.region, []).append(r)
    pairs = []
    for region, tiles in by_region.items():
        tiles_sorted = sorted(tiles, key=lambda t: t.date)
        for earlier, later in zip(tiles_sorted, tiles_sorted[1:]):
            pairs.append((earlier, later))
    return pairs