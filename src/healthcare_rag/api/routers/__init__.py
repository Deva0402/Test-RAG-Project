from healthcare_rag.api.routers.query import router as query_router
from healthcare_rag.api.routers.health import router as health_router
from healthcare_rag.api.routers.documents import router as documents_router

__all__=["query_router","health_router","documents_router"]
