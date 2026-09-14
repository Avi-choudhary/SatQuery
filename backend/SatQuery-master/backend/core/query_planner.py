from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryPlan:
    raw_query: str
    intent: str
    phenomenon: str
    operation: str
    requested_output: str
    target_concept: str
    specific_index: Optional[str] = None
    direction_filter: Optional[str] = None
    top_k: Optional[int] = None
    sort_by: Optional[str] = None
    criteria: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "intent": self.intent,
            "phenomenon": self.phenomenon,
            "operation": self.operation,
            "requested_output": self.requested_output,
            "target_concept": self.target_concept,
            "specific_index": self.specific_index,
            "direction_filter": self.direction_filter,
            "top_k": self.top_k,
            "sort_by": self.sort_by,
            "criteria": self.criteria,
        }


def _match_phrase(text: str, patterns: List[str]) -> bool:
    for pat in patterns:
        if re.search(rf"\b{re.escape(pat)}\b", text, re.IGNORECASE):
            return True
    return False


def _match_regex(text: str, regex_pattern: str) -> bool:
    return bool(re.search(regex_pattern, text, re.IGNORECASE))


def parse_query_plan(query: str) -> QueryPlan:
    q = (query or "").strip()
    ql = q.lower()

    if _match_regex(ql, r"\b(water\s*depth|depth\s*of\s*water|how\s*deep|bathymetry|water\s*level\s*in\s*m|depth\s*in\s*met(?:er|re)s?)\b"):
        return QueryPlan(
            raw_query=q,
            intent="unsupported_inquiry",
            phenomenon="unsupported_depth",
            operation="explain_limitation",
            requested_output="unsupported_explanation",
            target_concept="water depth / bathymetry",
        )

    if _match_regex(ql, r"\b(how\s*many\s*farmers?|farmer\s*counts?|how\s*many\s*people|how\s*many\s*workers?|who\s*works?|worker\s*counts?|farmer\s*labor|farming\s*labor)\b"):
        return QueryPlan(
            raw_query=q,
            intent="unsupported_inquiry",
            phenomenon="unsupported_human_count",
            operation="explain_limitation",
            requested_output="unsupported_explanation",
            target_concept="human farmer labor and headcount",
        )

    index_match = re.search(r"\b(can\s+you\s+(?:calculate|compute|determine|give|extract)|calculate|compute)\s+(ndvi|ndwi|ndbi)\b", ql)
    if index_match:
        idx_name = index_match.group(2).lower()
        return QueryPlan(
            raw_query=q,
            intent="capability_inquiry",
            phenomenon="specific_index",
            operation="index_capability",
            requested_output="capability_explanation",
            target_concept=f"{idx_name.upper()} spectral index",
            specific_index=idx_name,
        )

    alone_index = re.search(r"\b(ndvi|ndwi|ndbi)\b", ql)
    if alone_index and any(w in ql for w in ("can", "calculate", "compute", "value", "formula")):
        idx_name = alone_index.group(1).lower()
        return QueryPlan(
            raw_query=q,
            intent="capability_inquiry",
            phenomenon="specific_index",
            operation="index_capability",
            requested_output="capability_explanation",
            target_concept=f"{idx_name.upper()} spectral index",
            specific_index=idx_name,
        )

    if _match_regex(ql, r"\b(what\s*evidence|evidence\s*supports|how\s*do\s*you\s*know|evidence\s*quality|diagnostics|quality\s*factors)\b"):
        return QueryPlan(
            raw_query=q,
            intent="evidence_inquiry",
            phenomenon="evidence",
            operation="explain_evidence",
            requested_output="evidence_breakdown",
            target_concept="analytical evidence and diagnostics",
        )

    if _match_regex(ql, r"\b(why\s*did|why\s*might|why\s*would|why|what\s*caused|possible\s*causes?|cause\s*of|reasons?\s*for|explanation\s*for)\b"):
        return QueryPlan(
            raw_query=q,
            intent="attribution_inquiry",
            phenomenon="attribution",
            operation="hypothesis",
            requested_output="causal_hypotheses",
            target_concept="causal attribution and plausible change mechanisms",
        )

    direction = None
    if _match_regex(ql, r"\b(increase\s*or\s*decrease|gain\s*or\s*loss|expand\s*or\s*contract|expansion\s*or\s*reduction)\b"):
        direction = "both"
    elif _match_regex(ql, r"\b(increase|increased|expansion|expand|expanded|grow|growth|gain|gained|surge|spread)\b"):
        direction = "increase"
    elif _match_regex(ql, r"\b(decrease|decreased|reduction|reduce|reduced|loss|lost|shrink|shrunk|recession|recede|contract|contracted|drawdown)\b"):
        direction = "decrease"

    top_k = None
    top_k_match = re.search(r"\b(?:top|first|show(?:\s+the)?)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", ql)
    words_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    if top_k_match:
        val = top_k_match.group(1)
        top_k = words_to_num.get(val, int(val) if val.isdigit() else None)

    sort_by = "area_desc"
    if "smallest" in ql:
        sort_by = "area_asc"
    elif "highest magnitude" in ql or "most significant" in ql:
        sort_by = "magnitude_desc"

    is_explicit_ranking = bool(
        top_k is not None
        or _match_regex(ql, r"\b(which\s*(?:areas?|regions?)\s*changed\s*the\s*most|largest\s*changed|biggest\s*changed|greatest\s*changed|largest\s*region|largest\s*change)\b")
    )

    if is_explicit_ranking:
        is_singular_superlative = bool(
            re.search(r"\b(?:the|single)\s+(?:single\s+)?(?:largest|biggest|greatest)\s+(?:changed\s+)?(?:region|area|patch|cluster|zone)\b", ql)
            or re.search(r"\bwhere\s+is\s+the\s+(?:single\s+)?(?:largest|biggest|greatest)\b", ql)
            or "the largest changed region" in ql
            or "the single largest" in ql
            or "the largest region" in ql
        )
        if is_singular_superlative and top_k is None:
            top_k = 1
        return QueryPlan(
            raw_query=q,
            intent="ranking_inquiry",
            phenomenon="ranking",
            operation="rank",
            requested_output="top_ranked_regions",
            target_concept="spatial ranking of changed regions",
            top_k=top_k or 5,
            sort_by=sort_by,
        )

    if _match_regex(ql, r"\b(flood|flooding|inundat(?:ed|ion)?|submerged|deluge)\b"):
        return QueryPlan(
            raw_query=q,
            intent="change_inquiry",
            phenomenon="flooding",
            operation="spatial_localization",
            requested_output="direction_area_location",
            target_concept="flooding and surface water inundation",
            direction_filter=direction or "increase",
        )

    if _match_regex(ql, r"\b(water\s*bod(?:y|ies)|water\s*extent|water\s*surface|water\s*area|lake|lakes|river|rivers|reservoir|reservoirs|surface\s*water|wetness)\b"):
        return QueryPlan(
            raw_query=q,
            intent="change_inquiry",
            phenomenon="water",
            operation="temporal_direction" if direction else "detect_and_summarize",
            requested_output="direction_area_location",
            target_concept="water bodies and surface water extent",
            direction_filter=direction or "both",
        )

    if _match_regex(ql, r"\b(construction|excavat(?:ed|ion)?|earthworks?|ground\s*clearing|site\s*preparation|building\s*foundation)\b"):
        return QueryPlan(
            raw_query=q,
            intent="change_inquiry",
            phenomenon="construction",
            operation="spatial_localization",
            requested_output="direction_area_location",
            target_concept="construction activity and ground disturbance",
            direction_filter=direction or "increase",
        )

    if _match_regex(ql, r"\b(urbanisation|urbanization|urban\s*expansion|urban\s*footprint|urban\s*sprawl|urban\s*growth|city\s*expansion)\b"):
        return QueryPlan(
            raw_query=q,
            intent="change_inquiry",
            phenomenon="urbanisation",
            operation="temporal_direction" if direction else "detect_and_summarize",
            requested_output="direction_area_location",
            target_concept="urbanisation and urban footprint expansion",
            direction_filter=direction or "increase",
        )

    if _match_regex(ql, r"\b(built-?up(?:\s*area)?|new\s*buildings?|building\s*footprints?|structural\s*growth)\b"):
        return QueryPlan(
            raw_query=q,
            intent="change_inquiry",
            phenomenon="built_up",
            operation="temporal_direction" if direction else "detect_and_summarize",
            requested_output="direction_area_location",
            target_concept="built-up land and structural footprint",
            direction_filter=direction or "increase",
        )

    if _match_regex(ql, r"\b(farm|farming|agriculture|agricultural|cropland|farmland|crop|crops|cultivat(?:ed|ion)?|paddy)\b"):
        return QueryPlan(
            raw_query=q,
            intent="change_inquiry",
            phenomenon="agriculture",
            operation="temporal_direction" if direction else "detect_and_summarize",
            requested_output="direction_area_location",
            target_concept="agricultural and farming land cover signatures",
            direction_filter=direction or "both",
        )

    if _match_regex(ql, r"\b(vegetation|vegetated|green\s*cover|greenery|forest|forested|tree\s*canopy|canopy\s*cover|greenness|deforestation|reforestation)\b"):
        return QueryPlan(
            raw_query=q,
            intent="change_inquiry",
            phenomenon="vegetation",
            operation="temporal_direction" if direction else "detect_and_summarize",
            requested_output="direction_area_location",
            target_concept="vegetation vigor and green canopy cover",
            direction_filter=direction or "both",
        )

    if _match_regex(ql, r"\b(roads?|highways?|streets?|expressways?|runways?|linear\s*infrastructure|railways?|rail\s*tracks?)\b"):
        return QueryPlan(
            raw_query=q,
            intent="change_inquiry",
            phenomenon="infrastructure",
            operation="spatial_localization",
            requested_output="direction_area_location",
            target_concept="linear transportation infrastructure",
            direction_filter=direction or "both",
        )

    if _match_regex(ql, r"\b(brighten(?:ing)?|surface\s*brighten(?:ing)?|high\s*reflectance|lighter\s*surface)\b"):
        return QueryPlan(
            raw_query=q,
            intent="change_inquiry",
            phenomenon="brightening",
            operation="spatial_localization",
            requested_output="direction_area_location",
            target_concept="surface brightening",
            direction_filter="increase",
        )

    if _match_regex(ql, r"\b(darken(?:ing)?|surface\s*darken(?:ing)?|low\s*reflectance|darker\s*surface)\b"):
        return QueryPlan(
            raw_query=q,
            intent="change_inquiry",
            phenomenon="darkening",
            operation="spatial_localization",
            requested_output="direction_area_location",
            target_concept="surface darkening",
            direction_filter="decrease",
        )

    return QueryPlan(
        raw_query=q,
        intent="change_inquiry",
        phenomenon="general_change",
        operation="detect_and_summarize",
        requested_output="direct_answer_summary",
        target_concept="observable surface and land-cover change",
        direction_filter=direction,
        top_k=top_k,
        sort_by=sort_by,
    )
