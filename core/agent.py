from typing import List
from core.tracer import Tracer
from schemas.responses import SatQueryResponse
from services import vqa_service, ground_service, change_service

async def process_query(query: str, file_paths: List[str]) -> SatQueryResponse:
    tracer = Tracer()
    tracer.append_log(f"Received query: '{query}'")
    
    # Intent classification (parsing natural language to task)
    query_lower = query.lower()
    task = "UNKNOWN"
    
    # Lightweight LLM or keyword matcher to output a strict command
    if "new building" in query_lower or "change" in query_lower:
        task = "CHANGE_DETECTION"
    elif "where" in query_lower or "find" in query_lower:
        task = "GROUNDING"
    else:
        task = "VQA"
        
    tracer.append_log(f"step 1: classified task as {task}")
    
    visual_evidence = []
    text_answer = ""
    
    # Trigger the correct file in the services folder based on task
    if task == "CHANGE_DETECTION":
        text_answer, visual_evidence = await change_service.run_inference(query, file_paths, tracer)
    elif task == "GROUNDING":
        text_answer, visual_evidence = await ground_service.run_inference(query, file_paths, tracer)
    elif task == "VQA":
        text_answer, visual_evidence = await vqa_service.run_inference(query, file_paths, tracer)
    else:
        text_answer = "Could not determine the appropriate task for the query."
        
    return SatQueryResponse(
        text_answer=text_answer,
        visual_evidence=visual_evidence,
        execution_trace=tracer.get_trace()
    )
