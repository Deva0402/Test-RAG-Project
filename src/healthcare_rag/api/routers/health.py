from fastapi import APIRouter
from healthcare_rag.api.schemas.health import HealthResponse
import time

router=APIRouter(prefix="/health", tags=["health"])

@router.get("", response_model=HealthResponse)
async def health_check():
    try:
        from healthcare_rag.models.llama_cpp_model import LlamaCppModel
        from healthcare_rag.vectordb.chroma_store import ChromaStore

        model_loaded=True
        vector_db_connected=True

    except Exception:
        model_loaded=False
        vector_db_connected=False

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model_loaded,
        vector_db_connected=vector_db_connected
    )        