from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.query_planner import QueryPlan


@dataclass
class CapabilityAssessment:
    phenomenon: str
    target_concept: str
    can_measure: bool
    can_semantically_interpret: bool
    semantic_confidence: str  # "HIGH", "MODERATE", "LOW", "UNSUPPORTED"
    preferred_evidence_available: bool
    selected_method: str
    contributing_evidence_signals: List[str]
    semantic_status_label: str
    direct_answer_framing: str  # E.g. "confirmed_spectral", "candidate_proxy", "unsupported_explanation"
    limitations: List[str]
    unsupported_reason: Optional[str] = None
    closest_valid_analysis: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phenomenon": self.phenomenon,
            "target_concept": self.target_concept,
            "can_measure": self.can_measure,
            "can_semantically_interpret": self.can_semantically_interpret,
            "semantic_confidence": self.semantic_confidence,
            "preferred_evidence_available": self.preferred_evidence_available,
            "selected_method": self.selected_method,
            "contributing_evidence_signals": self.contributing_evidence_signals,
            "semantic_status_label": self.semantic_status_label,
            "direct_answer_framing": self.direct_answer_framing,
            "limitations": self.limitations,
            "unsupported_reason": self.unsupported_reason,
            "closest_valid_analysis": self.closest_valid_analysis,
        }


# Heuristic operating thresholds configured for this implementation
OPERATING_THRESHOLDS = {
    "delta_ndvi_significant": 0.15,
    "delta_ndwi_significant": 0.15,
    "delta_ndbi_significant": 0.15,
    "delta_brightness_significant": 0.06,
    "sar_delta_db_significant": 1.5,
    "compactness_built_up_candidate": 0.15,
    "compactness_linear_candidate": 0.06,
}


def evaluate_capability(
    plan: QueryPlan,
    modality: str,
    first_bands: Dict[str, Optional[int]],
    second_bands: Dict[str, Optional[int]],
    qa_available: bool = False,
    overlap_fraction: float = 1.0,
    shift_magnitude: float = 0.0,
) -> CapabilityAssessment:
    phenom = plan.phenomenon
    concept = plan.target_concept

    # 1. Physics / Sensor Unsupported Inquiries
    if phenom == "unsupported_depth":
        return CapabilityAssessment(
            phenomenon=phenom,
            target_concept=concept,
            can_measure=False,
            can_semantically_interpret=False,
            semantic_confidence="UNSUPPORTED",
            preferred_evidence_available=False,
            selected_method="unsupported_physics",
            contributing_evidence_signals=[],
            semantic_status_label="unsupported_depth_measurement",
            direct_answer_framing="unsupported_explanation",
            limitations=[
                "Two-dimensional satellite optical and SAR sensors measure surface reflectance and backscatter; they cannot measure water depth, volume, or underwater bathymetry."
            ],
            unsupported_reason="Water depth requires bathymetric lidar, sonar sounding, or calibrated hydrodynamic models, which are absent in standard 2D satellite rasters.",
            closest_valid_analysis="Two-dimensional surface water extent (area in hectares, shoreline location, and surface water boundaries) can be measured instead.",
        )

    if phenom == "unsupported_human_count":
        return CapabilityAssessment(
            phenomenon=phenom,
            target_concept=concept,
            can_measure=False,
            can_semantically_interpret=False,
            semantic_confidence="UNSUPPORTED",
            preferred_evidence_available=False,
            selected_method="unsupported_social",
            contributing_evidence_signals=[],
            semantic_status_label="unsupported_human_measurement",
            direct_answer_framing="unsupported_explanation",
            limitations=[
                "Standard satellite imagery does not observe individual human presence, labor hours, or agricultural workforce headcount."
            ],
            unsupported_reason="Satellite sensors measure land surface properties, not human individuals or labor activity.",
            closest_valid_analysis="Observable vegetative greenness vigor, crop parcel boundaries, and field surface reflectance changes can be quantified instead.",
        )

    # 2. Specific Spectral Index Capability
    if phenom == "specific_index":
        req_bands: Tuple[str, ...] = ()
        idx = (plan.specific_index or "").lower()
        if idx == "ndvi":
            req_bands = ("red", "nir")
        elif idx == "ndwi":
            req_bands = ("green", "nir")
        elif idx == "ndbi":
            req_bands = ("nir", "swir1")

        if modality != "optical":
            return CapabilityAssessment(
                phenomenon=phenom,
                target_concept=concept,
                can_measure=False,
                can_semantically_interpret=False,
                semantic_confidence="UNSUPPORTED",
                preferred_evidence_available=False,
                selected_method="sar_index_unsupported",
                contributing_evidence_signals=[],
                semantic_status_label="sar_cannot_calculate_optical_index",
                direct_answer_framing="unsupported_explanation",
                limitations=[
                    f"The supplied imagery is SAR (radar backscatter), which operates in microwave frequencies and lacks optical spectral bands required for {idx.upper()}."
                ],
                unsupported_reason=f"Calculating {idx.upper()} requires optical multispectral bands {list(req_bands)}, which do not exist in SAR data.",
                closest_valid_analysis="Radar backscatter log-ratio and amplitude variation analysis are available for this SAR pair.",
            )

        has_req = all(
            first_bands.get(b) is not None and second_bands.get(b) is not None
            for b in req_bands
        )
        if has_req:
            return CapabilityAssessment(
                phenomenon=phenom,
                target_concept=concept,
                can_measure=True,
                can_semantically_interpret=True,
                semantic_confidence="HIGH",
                preferred_evidence_available=True,
                selected_method=f"{idx}_difference",
                contributing_evidence_signals=[f"{idx}_difference", "spectral_bands"],
                semantic_status_label=f"{idx}_supported",
                direct_answer_framing="index_supported",
                limitations=[
                    f"{idx.upper()} is computed from normalized band differences at sensor spatial resolution."
                ],
            )
        else:
            missing_1 = [b for b in req_bands if first_bands.get(b) is None]
            missing_2 = [b for b in req_bands if second_bands.get(b) is None]
            all_missing = sorted(list(set(missing_1 + missing_2)))
            return CapabilityAssessment(
                phenomenon=phenom,
                target_concept=concept,
                can_measure=False,
                can_semantically_interpret=False,
                semantic_confidence="UNSUPPORTED",
                preferred_evidence_available=False,
                selected_method="missing_index_bands",
                contributing_evidence_signals=[],
                semantic_status_label=f"{idx}_missing_bands",
                direct_answer_framing="unsupported_explanation",
                limitations=[
                    f"Calculating {idx.upper()} requires {list(req_bands)} bands, but {all_missing} are missing from the input imagery."
                ],
                unsupported_reason=f"Required spectral band(s) {all_missing} were not embedded in the uploaded GeoTIFF.",
                closest_valid_analysis="Multiband Change Vector Analysis (CVA) across available visible bands is available instead.",
            )

    # 3. Water Phenomena (water extent & flooding)
    if phenom in ("water", "flooding"):
        has_ndwi_bands = bool(
            modality == "optical"
            and first_bands.get("green") is not None
            and first_bands.get("nir") is not None
            and second_bands.get("green") is not None
            and second_bands.get("nir") is not None
        )

        if has_ndwi_bands:
            signals = ["optical_cva", "ndwi_spectral_evidence", "surface_darkening_metric"]
            return CapabilityAssessment(
                phenomenon=phenom,
                target_concept=concept,
                can_measure=True,
                can_semantically_interpret=True,
                semantic_confidence="HIGH" if shift_magnitude < 1.5 else "MODERATE",
                preferred_evidence_available=True,
                selected_method="optical_ndwi_fusion",
                contributing_evidence_signals=signals,
                semantic_status_label="water_extent_supported" if phenom == "water" else "candidate_inundation_signature",
                direct_answer_framing="water_supported",
                limitations=[
                    "Water extent is derived from 2D surface reflectance and NDWI; water depth and underwater volume cannot be determined."
                ],
            )

        if modality == "sar":
            signals = ["sar_log_ratio", "specular_backscatter_decrease"]
            return CapabilityAssessment(
                phenomenon=phenom,
                target_concept=concept,
                can_measure=True,
                can_semantically_interpret=True,
                semantic_confidence="MODERATE",
                preferred_evidence_available=True,
                selected_method="sar_water_evidence",
                contributing_evidence_signals=signals,
                semantic_status_label="radar_water_extent_proxy" if phenom == "water" else "radar_candidate_inundation",
                direct_answer_framing="water_sar_proxy",
                limitations=[
                    "Smooth standing water causes specular reflection in radar resulting in backscatter reduction; other smooth flat surfaces (e.g. flat paved surfaces or airport runways) may produce similar backscatter reductions.",
                    "Water depth and bathymetry cannot be measured from SAR backscatter."
                ],
            )

        # Optical RGB-only (missing NIR)
        return CapabilityAssessment(
            phenomenon=phenom,
            target_concept=concept,
            can_measure=True,
            can_semantically_interpret=False,
            semantic_confidence="LOW",
            preferred_evidence_available=False,
            selected_method="rgb_darkening_proxy",
            contributing_evidence_signals=["optical_cva", "surface_darkening_signal"],
            semantic_status_label="unconfirmed_water_proxy_rgb_only",
            direct_answer_framing="water_rgb_proxy",
            limitations=[
                "NIR band is unavailable in this optical imagery, preventing direct NDWI water index calculation.",
                "Surface darkening is detected, but visible darkening alone cannot definitively confirm water body extent because cloud shadows, dark soil, or asphalt can produce identical visible darkening."
            ],
            closest_valid_analysis="Detected surface-darkening regions and general spatial change are quantified, but water attribution remains candidate/uncertain.",
        )

    # 4. Vegetation / Greenery
    if phenom == "vegetation":
        has_ndvi_bands = bool(
            modality == "optical"
            and first_bands.get("red") is not None
            and first_bands.get("nir") is not None
            and second_bands.get("red") is not None
            and second_bands.get("nir") is not None
        )

        if has_ndvi_bands:
            signals = ["optical_cva", "ndvi_spectral_evidence", "vegetative_vigor_profile"]
            return CapabilityAssessment(
                phenomenon=phenom,
                target_concept=concept,
                can_measure=True,
                can_semantically_interpret=True,
                semantic_confidence="HIGH" if shift_magnitude < 1.5 else "MODERATE",
                preferred_evidence_available=True,
                selected_method="optical_ndvi_fusion",
                contributing_evidence_signals=signals,
                semantic_status_label="vegetation_greenness_change",
                direct_answer_framing="vegetation_supported",
                limitations=[
                    "NDVI measures vegetative greenness vigor and photosynthetic activity; it does not measure dry biomass in kg or species-level taxonomy."
                ],
            )

        if modality == "sar":
            return CapabilityAssessment(
                phenomenon=phenom,
                target_concept=concept,
                can_measure=True,
                can_semantically_interpret=False,
                semantic_confidence="LOW",
                preferred_evidence_available=False,
                selected_method="sar_volume_scattering_proxy",
                contributing_evidence_signals=["sar_log_ratio", "backscatter_variation"],
                semantic_status_label="sar_vegetation_unsupported_direct",
                direct_answer_framing="vegetation_sar_proxy",
                limitations=[
                    "SAR radar backscatter is sensitive to surface roughness and canopy dielectric properties, but cannot calculate NDVI or direct photosynthetic greenness."
                ],
                closest_valid_analysis="General SAR backscatter variation is reported; optical Red+NIR imagery is recommended for vegetative index quantification.",
            )

        # Optical RGB-only (missing NIR)
        return CapabilityAssessment(
            phenomenon=phenom,
            target_concept=concept,
            can_measure=True,
            can_semantically_interpret=False,
            semantic_confidence="LOW",
            preferred_evidence_available=False,
            selected_method="rgb_visible_chromatic_proxy",
            contributing_evidence_signals=["optical_cva", "visible_chromatic_change"],
            semantic_status_label="unconfirmed_vegetation_proxy_rgb_only",
            direct_answer_framing="vegetation_rgb_proxy",
            limitations=[
                "NIR band is absent in this dataset, so NDVI cannot be calculated.",
                "Visible RGB bands exhibit localized chromatic variation, but vegetative vigor quantification cannot be confirmed without NIR."
            ],
            closest_valid_analysis="General visible spectral change is quantified; Red+NIR imagery is required for verified NDVI quantification.",
        )

    # 5. Agriculture / Farming
    if phenom == "agriculture":
        has_ndvi_bands = bool(
            modality == "optical"
            and first_bands.get("red") is not None
            and first_bands.get("nir") is not None
            and second_bands.get("red") is not None
            and second_bands.get("nir") is not None
        )

        signals = ["optical_cva", "field_parcel_geometry"]
        if has_ndvi_bands:
            signals.append("crop_phenology_ndvi")

        return CapabilityAssessment(
            phenomenon=phenom,
            target_concept=concept,
            can_measure=True,
            can_semantically_interpret=True,
            semantic_confidence="MODERATE",
            preferred_evidence_available=has_ndvi_bands,
            selected_method="agricultural_land_cover_fusion",
            contributing_evidence_signals=signals,
            semantic_status_label="agricultural_land_signature_change",
            direct_answer_framing="agriculture_supported",
            limitations=[
                "SatQuery observes agricultural land surface signatures, crop vigor proxies, and field-pattern variations.",
                "Satellite imagery cannot measure direct human farmer activity, labor hours, crop yield in metric tonnes, or economic decisions."
            ],
        )

    # 6. Urbanisation / Built-up / Construction
    if phenom in ("urbanisation", "built_up", "construction"):
        has_swir_bands = bool(
            modality == "optical"
            and first_bands.get("swir1") is not None
            and first_bands.get("nir") is not None
            and second_bands.get("swir1") is not None
            and second_bands.get("nir") is not None
        )

        signals = ["optical_cva", "region_compactness"]
        if has_swir_bands:
            signals.append("ndbi_spectral_evidence")
        else:
            signals.append("surface_brightening_signal")

        if modality == "sar":
            signals = ["sar_log_ratio", "double_bounce_backscatter_increase", "region_compactness"]

        confidence = "MODERATE" if (has_swir_bands or modality == "sar") else "MODERATE"
        return CapabilityAssessment(
            phenomenon=phenom,
            target_concept=concept,
            can_measure=True,
            can_semantically_interpret=True,
            semantic_confidence=confidence,
            preferred_evidence_available=has_swir_bands or modality == "sar",
            selected_method="built_up_multievidence_fusion",
            contributing_evidence_signals=signals,
            semantic_status_label="candidate_built_up_expansion" if phenom != "construction" else "candidate_construction_disturbance",
            direct_answer_framing="urban_candidate_proxy",
            limitations=[
                "Classified as candidate built-up / construction expansion signatures based on multi-evidence fusion.",
                "Surface brightening, high compactness, and radar backscatter surge serve as candidate proxies; individual 3D building counts, interior occupancy, and structural height are not measured."
            ],
        )

    # 7. Linear Transportation Infrastructure / Roads
    if phenom == "infrastructure":
        signals = ["optical_cva" if modality == "optical" else "sar_log_ratio", "linear_geometry_filter", "low_compactness"]
        return CapabilityAssessment(
            phenomenon=phenom,
            target_concept=concept,
            can_measure=True,
            can_semantically_interpret=True,
            semantic_confidence="MODERATE",
            preferred_evidence_available=True,
            selected_method="linear_infrastructure_geometry_fusion",
            contributing_evidence_signals=signals,
            semantic_status_label="candidate_linear_infrastructure",
            direct_answer_framing="infrastructure_candidate_proxy",
            limitations=[
                "Linear corridor geometry (low compactness, high elongation) serves as candidate infrastructure evidence only.",
                "Ground clearing, drainage channels, or firebreaks can exhibit similar linear shapes and cannot be definitively distinguished from roads without external GIS vector layers.",
                "Traffic volume and vehicle counts are not measured."
            ],
        )

    # 8. Brightening / Darkening
    if phenom in ("brightening", "darkening"):
        signals = ["optical_cva", f"delta_brightness_{phenom}"]
        return CapabilityAssessment(
            phenomenon=phenom,
            target_concept=concept,
            can_measure=True,
            can_semantically_interpret=True,
            semantic_confidence="HIGH",
            preferred_evidence_available=True,
            selected_method=f"surface_{phenom}_analysis",
            contributing_evidence_signals=signals,
            semantic_status_label=f"surface_{phenom}",
            direct_answer_framing="radiometric_signal_direct",
            limitations=[
                f"Measures physical visible surface {phenom} across the scene footprint.",
                "Plausible underlying causes (wetting, drying, clearing, shadows) are evaluated as hypotheses."
            ],
        )

    # 9. Ranking / Region Query
    if phenom == "ranking":
        signals = ["polygon_area_ranking", "centroid_coordinates", "bounding_boxes"]
        return CapabilityAssessment(
            phenomenon=phenom,
            target_concept=concept,
            can_measure=True,
            can_semantically_interpret=True,
            semantic_confidence="HIGH",
            preferred_evidence_available=True,
            selected_method="gis_spatial_ranking",
            contributing_evidence_signals=signals,
            semantic_status_label="ranked_change_regions",
            direct_answer_framing="ranking_direct",
            limitations=[
                "Rankings are strictly based on polygonized geometric area in projected map coordinates."
            ],
        )

    # 10. Evidence / Audit inquiry
    if phenom == "evidence":
        signals = ["registration_residual", "pif_normalization", "cva_distribution", "reliability_gate"]
        return CapabilityAssessment(
            phenomenon=phenom,
            target_concept=concept,
            can_measure=True,
            can_semantically_interpret=True,
            semantic_confidence="HIGH",
            preferred_evidence_available=True,
            selected_method="audit_evidence_reporting",
            contributing_evidence_signals=signals,
            semantic_status_label="audit_evidence",
            direct_answer_framing="evidence_direct",
            limitations=[
                "Evidence quality metrics represent algorithmic convergence and sensor data consistency diagnostics."
            ],
        )

    # 11. Causal Attribution ("Why?")
    if phenom == "attribution":
        signals = ["change_type_signatures", "spectral_deltas", "hypothesis_engine"]
        return CapabilityAssessment(
            phenomenon=phenom,
            target_concept=concept,
            can_measure=True,
            can_semantically_interpret=True,
            semantic_confidence="MODERATE",
            preferred_evidence_available=True,
            selected_method="causal_hypothesis_synthesis",
            contributing_evidence_signals=signals,
            semantic_status_label="causal_hypotheses",
            direct_answer_framing="attribution_hypotheses",
            limitations=[
                "Causal explanations are plausible scientific hypotheses inferred from radiometric and geometric signatures.",
                "Definitive causal certainty requires in-situ ground inspection or historical administrative records."
            ],
        )

    # 12. General change default
    signals = ["optical_cva" if modality == "optical" else "sar_log_ratio", "pif_normalization", "scene_thresholding"]
    return CapabilityAssessment(
        phenomenon="general_change",
        target_concept="general surface change",
        can_measure=True,
        can_semantically_interpret=True,
        semantic_confidence="HIGH" if shift_magnitude < 1.5 else "MODERATE",
        preferred_evidence_available=True,
        selected_method="optical_cva_general" if modality == "optical" else "sar_log_ratio_general",
        contributing_evidence_signals=signals,
        semantic_status_label="general_surface_change",
        direct_answer_framing="general_summary",
        limitations=[
            "Analysis evaluates observable surface spectral change within sensor spatial resolution bounds."
        ],
    )
