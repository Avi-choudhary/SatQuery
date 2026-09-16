from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class SatQueryResponse(BaseModel):
    text_answer: str = Field(
        ...,
        description="Natural-language answer produced by the selected specialist.",
    )

    visual_evidence: List[Union[str, List[float], Dict[str, Any]]] = Field(
        default_factory=list,
        description="Paths, URLs, coordinate bounding boxes, or GeoJSON features/FeatureCollections.",
    )

    execution_trace: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent execution steps, diagnostics and summary.",
    )

    trace_log: Optional[List[str]] = Field(
        default=None,
        description="Flat chronological list of execution steps for UI rendering.",
    )

    conversation_id: Optional[str] = Field(
        default=None,
        description="Persisted conversation ID for this turn.",
    )

    band_contract: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Authoritative band capability contract for loaded imagery.",
    )
