import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import rasterio
from rasterio.enums import ColorInterp

def detect_raster_bands(raster_input: Union[str, Path, rasterio.io.DatasetReader]) -> Dict[str, Any]:
    """
    Authoritatively inspects a raster container to detect physical spectral bands.
    Does NOT infer from filename; inspects band count, descriptions, color interpretation,
    and band metadata tags.
    """
    should_close = False
    if isinstance(raster_input, (str, Path)):
        if not os.path.exists(str(raster_input)):
            return {
                "filename": os.path.basename(str(raster_input)),
                "band_count": 0,
                "descriptions": [],
                "colorinterp": [],
                "bands": {"red": None, "green": None, "blue": None, "nir": None, "swir1": None},
                "has_rgb": False,
                "has_nir": False,
                "has_swir1": False,
                "nir_band_index": None,
                "red_band_index": None,
                "green_band_index": None,
                "blue_band_index": None,
            }
        dataset = rasterio.open(str(raster_input))
        should_close = True
    else:
        dataset = raster_input

    try:
        count = dataset.count
        raw_descriptions = list(dataset.descriptions) if dataset.descriptions else [None] * count
        descriptions = [str(d).strip() if d else "" for d in raw_descriptions]
        
        # Color interpretation
        color_interps = []
        try:
            for ci in dataset.colorinterp:
                name = ci.name.lower() if hasattr(ci, "name") else str(ci).lower()
                color_interps.append(name)
        except Exception:
            color_interps = ["undefined"] * count

        red_idx: Optional[int] = None
        green_idx: Optional[int] = None
        blue_idx: Optional[int] = None
        nir_idx: Optional[int] = None
        swir1_idx: Optional[int] = None

        # 1. Match based on explicit descriptions (e.g. Sentinel-2: B4=Red, B3=Green, B2=Blue, B8=NIR)
        for i, desc in enumerate(descriptions, start=1):
            d_clean = desc.lower().replace(" ", "").replace("_", "").replace("-", "")
            if not d_clean:
                continue
            
            # NIR candidates
            if d_clean in ("b8", "b8a", "nir", "nearinfrared", "band8", "band8a"):
                nir_idx = i
            # SWIR1 candidates
            elif d_clean in ("b11", "swir", "swir1", "band11"):
                swir1_idx = i
            # Red candidates
            elif d_clean in ("b4", "red", "band4"):
                red_idx = i
            # Green candidates
            elif d_clean in ("b3", "green", "band3"):
                green_idx = i
            # Blue candidates
            elif d_clean in ("b2", "blue", "band2"):
                blue_idx = i

        # 2. Match based on ColorInterp if not already found
        if not red_idx or not green_idx or not blue_idx:
            for i, ci in enumerate(color_interps, start=1):
                if ci == "red" and not red_idx:
                    red_idx = i
                elif ci == "green" and not green_idx:
                    green_idx = i
                elif ci == "blue" and not blue_idx:
                    blue_idx = i

        # 3. Fallback conventional mappings based on band count
        if count == 3:
            # Standard RGB 3-band raster
            if not red_idx:
                red_idx = 1
            if not green_idx:
                green_idx = 2
            if not blue_idx:
                blue_idx = 3
        elif count == 4:
            # Common 4-band satellite stack (e.g. Sentinel-2 B4, B3, B2, B8 or B2, B3, B4, B8)
            if not red_idx and not green_idx and not blue_idx and not nir_idx:
                # Default standard Sentinel-2 10m L2A band ordering: [B4, B3, B2, B8]
                red_idx = 1
                green_idx = 2
                blue_idx = 3
                nir_idx = 4
            elif (red_idx and green_idx and blue_idx) and not nir_idx:
                # If red, green, blue take 3 of the 4 bands, the remaining 4th band is NIR
                used = {red_idx, green_idx, blue_idx}
                remaining = [b for b in range(1, 5) if b not in used]
                if remaining:
                    nir_idx = remaining[0]

        has_rgb = bool(red_idx is not None and green_idx is not None and blue_idx is not None)
        has_nir = bool(nir_idx is not None)
        has_swir1 = bool(swir1_idx is not None)

        filename = os.path.basename(dataset.name) if hasattr(dataset, "name") else "unknown"

        return {
            "filename": filename,
            "band_count": count,
            "descriptions": descriptions,
            "colorinterp": color_interps,
            "bands": {
                "red": red_idx,
                "green": green_idx,
                "blue": blue_idx,
                "nir": nir_idx,
                "swir1": swir1_idx,
            },
            "has_rgb": has_rgb,
            "has_nir": has_nir,
            "has_swir1": has_swir1,
            "nir_band_index": nir_idx,
            "red_band_index": red_idx,
            "green_band_index": green_idx,
            "blue_band_index": blue_idx,
        }
    finally:
        if should_close:
            dataset.close()


def build_band_capability_contract(
    t1_path: Union[str, Path],
    t2_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Builds the authoritative band contract and analysis capabilities for single or
    bi-temporal satellite scenes.
    """
    t1_info = detect_raster_bands(t1_path)
    t2_info = detect_raster_bands(t2_path) if t2_path else None

    is_bitemporal = t2_info is not None

    t1_has_nir = bool(t1_info["has_nir"])
    t2_has_nir = bool(t2_info["has_nir"]) if t2_info else False

    t1_has_rgb = bool(t1_info["has_rgb"])
    t2_has_rgb = bool(t2_info["has_rgb"]) if t2_info else False

    common_rgb = bool(t1_has_rgb and (t2_has_rgb if is_bitemporal else True))
    common_nir = bool(t1_has_nir and (t2_has_nir if is_bitemporal else False))

    capabilities = {
        # T1 Capabilities
        "t1_true_color": t1_has_rgb,
        "t1_false_color_nir": t1_has_nir,
        "t1_ndvi": bool(t1_has_rgb and t1_has_nir),
        
        # T2 Capabilities (if present)
        "t2_true_color": t2_has_rgb if is_bitemporal else False,
        "t2_false_color_nir": t2_has_nir if is_bitemporal else False,
        "t2_ndvi": bool(t2_has_rgb and t2_has_nir) if is_bitemporal else False,

        # Multi-temporal joint capabilities
        "bitemporal_cva": common_rgb,
        "bitemporal_ndvi": common_nir,
        "bitemporal_false_color_nir": common_nir,
    }

    # Summary indicator text (e.g. T1: RGB ✓ | NIR ✓ | NDVI ✓)
    def _summary_str(info: Dict[str, Any]) -> str:
        rgb_c = "✓" if info["has_rgb"] else "✗"
        nir_c = "✓" if info["has_nir"] else "✗"
        ndvi_c = "✓" if (info["has_rgb"] and info["has_nir"]) else "✗"
        return f"RGB {rgb_c} | NIR {nir_c} | NDVI {ndvi_c}"

    t1_summary = f"T1: {_summary_str(t1_info)}"
    t2_summary = f"T2: {_summary_str(t2_info)}" if t2_info else None

    return {
        "mode": "bi-temporal" if is_bitemporal else "single",
        "t1": t1_info,
        "t2": t2_info,
        "joint": {
            "common_rgb": common_rgb,
            "common_nir": common_nir,
            "bitemporal_ndvi_available": common_nir,
            "bitemporal_ndwi_available": bool(
                common_nir and t1_info.get("bands", {}).get("green") and (t2_info.get("bands", {}).get("green") if t2_info else False)
            ),
        },
        "capabilities": capabilities,
        "indicators": {
            "t1": t1_summary,
            "t2": t2_summary,
        }
    }


def format_band_contract_markdown(contract: Dict[str, Any]) -> str:
    """
    Renders an authoritative, human-readable GFM Markdown report describing
    spectral band availability and analytical index feasibility.
    """
    from utils.markdown_formatter import format_markdown_table

    t1 = contract.get("t1", {})
    t2 = contract.get("t2")
    is_bitemporal = contract.get("mode") == "bi-temporal" and t2 is not None

    t1_name = t1.get("filename", "T1")
    t1_cnt = t1.get("band_count", 0)
    t1_has_nir = t1.get("has_nir", False)
    t1_nir_idx = t1.get("nir_band_index")
    t1_nir_str = f"✅ Available (Band {t1_nir_idx})" if t1_has_nir else "❌ Absent"
    t1_bands_str = "Red, Green, Blue, NIR" if t1_has_nir else "Red, Green, Blue"

    headers = ["Acquisition / View", "Band Count", "Spectral Channels", "NIR Band Status", "Single-Date Capability"]
    rows = [
        [f"**T1 ('{t1_name}')**", str(t1_cnt), t1_bands_str, t1_nir_str, "True-Color RGB" + (", False-Color NIR, NDVI" if t1_has_nir else "")]
    ]

    if is_bitemporal:
        t2_name = t2.get("filename", "T2")
        t2_cnt = t2.get("band_count", 0)
        t2_has_nir = t2.get("has_nir", False)
        t2_nir_idx = t2.get("nir_band_index")
        t2_nir_str = f"✅ Available (Band {t2_nir_idx})" if t2_has_nir else "❌ Absent"
        t2_bands_str = "Red, Green, Blue, NIR" if t2_has_nir else "Red, Green, Blue"

        rows.append([f"**T2 ('{t2_name}')**", str(t2_cnt), t2_bands_str, t2_nir_str, "True-Color RGB" + (", False-Color NIR, NDVI" if t2_has_nir else "")])
        
        common_nir = contract.get("joint", {}).get("common_nir", False)
        pair_nir_status = "✅ Complete Across Dates" if common_nir else "❌ Incomplete Across Dates"
        pair_cap = "Bitemporal CVA, ΔNDVI, False-Color NIR" if common_nir else "Bitemporal CVA (ΔNDVI Unavailable)"
        rows.append(["**Bi-temporal Pair**", "—", "Common: Visible RGB" + (", NIR" if common_nir else ""), pair_nir_status, pair_cap])

    table_md = format_markdown_table(
        headers=headers,
        rows=rows,
        alignments=["left", "center", "left", "center", "left"],
    )

    if is_bitemporal:
        common_nir = contract.get("joint", {}).get("common_nir", False)
        t1_has_nir = t1.get("has_nir", False)
        t2_has_nir = t2.get("has_nir", False)

        if common_nir:
            header_lead = "Both T1 and T2 contain visible and Near-Infrared (NIR) bands. Multi-temporal change detection, false-color NIR composites, and bi-temporal ΔNDVI are fully supported."
        elif t1_has_nir and not t2_has_nir:
            header_lead = f"Near-Infrared (NIR) is available in T1 ('{t1_name}', Band {t1.get('nir_band_index', 4)}), but absent in T2 ('{t2.get('filename', 'T2')}'). Bi-temporal vegetation index change (ΔNDVI) cannot be calculated for this pair, though single-date NDVI and false-color NIR are supported on T1."
        elif not t1_has_nir and t2_has_nir:
            header_lead = f"Near-Infrared (NIR) is available in T2 ('{t2.get('filename', 'T2')}'), but absent in T1 ('{t1_name}'). Bi-temporal vegetation index change (ΔNDVI) cannot be calculated for this pair."
        else:
            header_lead = "Both T1 and T2 contain standard 3-band RGB imagery. Visible Change Vector Analysis (CVA) is supported, but NIR and NDVI capabilities are unavailable."
    else:
        if t1_has_nir:
            header_lead = f"T1 ('{t1_name}') contains 4 spectral bands including Near-Infrared (NIR Band {t1.get('nir_band_index', 4)}). True-color RGB, false-color NIR, and NDVI calculation are supported."
        else:
            header_lead = f"T1 ('{t1_name}') contains 3 visible bands (RGB). True-color RGB visualization is supported. NIR and NDVI calculations require a Near-Infrared band."

    return (
        f"{header_lead}{table_md}"
        "### 🔬 Spectral Band Inspection Details\n"
        f"- **T1 Spectral Composition:** {t1_bands_str} ({t1_cnt} bands detected)\n"
        + (f"- **T2 Spectral Composition:** {t2_bands_str} ({t2_cnt} bands detected)\n" if is_bitemporal else "")
        + "\n### 💡 Analytical Capabilities\n"
        + f"- **True-Color RGB (Red, Green, Blue):** Supported\n"
        + f"- **False-Color NIR (NIR, Red, Green):** {'Supported on T1' if t1_has_nir else 'Unavailable'}\n"
        + f"- **NDVI Calculation ((NIR - Red) / (NIR + Red)):** {'Single-date supported on T1; bi-temporal ΔNDVI requires NIR in T2' if (is_bitemporal and t1_has_nir and not t2_has_nir) else ('Fully supported' if (t1_has_nir and (t2_has_nir if is_bitemporal else True)) else 'Unavailable')}"
    )
