from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool=Field(..., description="Whether LLM is loaded")
    vector_db_connected: bool=Field(..., description="Whether vector DB is connected")
