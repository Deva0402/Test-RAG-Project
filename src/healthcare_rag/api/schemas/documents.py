from pydantic import BaseModel,Field
from typing import Optional, Dict

class DocumentIngestRequest(BaseModel):
    
    file_path: str=Field(..., description="Path to the file or directory to ingest")
    
    metadata: Optional[Dict]=Field(default_factory=dict,description="Additional metadata")

class DocumentIngestResponse(BaseModel):
    success: bool=Field(..., description="Whether ingestion was successful")
    document_id: str=Field(..., description="ID of the ingested document")
    chunks_created: int=Field(..., description="Number of chunks created")
    message: str=Field(..., description="Status message")    