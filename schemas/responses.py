from pydantic import BaseModel
from typing import List, Dict, Any, Union

class SatQueryResponse(BaseModel):
    text_answer: str
    visual_evidence: List[Union[str, List[float]]]  # mask URLs or bounding box coords
    execution_trace: Dict[str, Any]
