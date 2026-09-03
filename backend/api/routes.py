import os
import shutil
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from schemas.responses import SatQueryResponse
from core.agent import process_query

router = APIRouter()

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/satquery", response_model=SatQueryResponse)
async def handle_satquery(
    query: str = Form(...),
    files: List[UploadFile] = File(...)
):
    saved_file_paths = []
    try:
        # Save massive GeoTIFFs to a temporary local folder so RAM doesn't overload
        for file in files:
            file_location = os.path.join(TEMP_DIR, file.filename)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_file_paths.append(file_location)
            
        # Pass file paths to the agent
        response = await process_query(query, saved_file_paths)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pass
        # Clean up code could go here, or handled by a separate background task
