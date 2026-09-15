from fastapi import APIRouter,HTTPException
from healthcare_rag.api.schemas.documents import DocumentIngestRequest, DocumentIngestResponse

from healthcare_rag.data.ingestion_pipeline import IngestionPipeline
from healthcare_rag.vectordb.chroma_store import ChromaStore
import uuid

router=APIRouter(prefix="/api/v1/documents", tags=["Documents"])

ingestion_pipeline=IngestionPipeline()
vector_store=ChromaStore()

@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_document(request: DocumentIngestRequest):
    try:
        document={
            "source":request.source,
            "text": request.text,
            "metadata": request.metadata
        }

        chunks=Ingestion_pipeline.ingest(request.text)

        chunk_ids=[]
        for chunk in chunks:
            chunk_id=str(uuid.uuid4)
            chunk_ids.append(chunk_id)

            vector_store.add_documents(
                text=chunk["text"],
                metadata={
                    **chunk["metadata"],
                    "chunk_id": chunk_id,
                    "source": request.source
                }
            )
        return DocumentIngestResponse(
            success=True,
            document_id=str(uuid.uuid4()),
            chunks_created=len(chunks),
            message=f"successfully ingested document with{len(chunks)} chunks"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed:{str(e)}")    