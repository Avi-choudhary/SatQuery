from pydantic import BaseModel
from typing import List, Dict, Any, Union, Optional

class SatQueryResponse(BaseModel):
    text_answer: str
    visual_evidence: List[Union[str, List[float], Dict[str, Any]]]  # mask URLs, bounding box coords, or GeoJSON features
    execution_trace: Dict[str, Any]  # contains 'steps', 'logs', and 'summary'
    trace_log: Optional[List[str]] = None  # direct list alias for convenience
