from typing import List, Tuple
from core.tracer import Tracer

async def run_inference(query: str, file_paths: List[str], tracer: Tracer) -> Tuple[str, List]:
    # Wrapper calling Member 4's bi-temporal model
    tracer.append_log("step 2: routed to ChangeFormer model")
    
    # Import model inference function here and pass cleaned image path
    
    # Mock return
    text_answer = "Change detected."
    visual_evidence = ["http://mask-url.example.com"]
    tracer.append_log("step 3: inference complete. confidence: 89%")
    
    return text_answer, visual_evidence
