from pydantic import BaseModel

class SatQueryRequest(BaseModel):
    query: str
    # Note: UploadFile can't be strictly validated in a pure Pydantic BaseModel for FastAPI 
    # the way standard types are, but we define the logical expectation here.
    # In routes.py, we will inject these as Form() and File().
