from unittest import result

from fastapi import APIRouter, HTTPException
from sklearn import pipeline
from healthcare_rag.api.schemas.query import QueryRequest,QueryResponse
from healthcare_rag.rag.pipeline import RAGPipeline
import time

router=APIRouter(prefix="/api/v1/query", tags=["Query"])

rag_pipeline=RAGPipeline(retriever_type="vector")

@router.post("", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    start_time=time.time()
    try:
        result=rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        processing_time=time.time()-start_time

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            retrieved_count=result["retrieved_count"],
            processing_time=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Query processing failed: {str(e)}")