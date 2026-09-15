

from pydantic import BaseModel, Field
from typing import List,Optional

class QueryRequest(BaseModel):
    query: str=Field(..., description="user query text",min_length=1)
    top_k: Optional[int]=Field(default=5,ge=1,le=20,description="Number of chunks to retrieve")
    use_reranker: Optional[bool]=Field(default=5)

class QueryResponse(BaseModel):
    answer:str=Field(..., description="Generated answer")
    sources: List[str]=Field(default_factory=list,description="source documents")
    retrieved_count: int=Field(..., description="Number of documents retrieved")
    processing_time: float=Field(..., description="processing time in seconds")