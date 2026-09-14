from pathlib import Path
from typing import Optional


from healthcare_rag.data.ingestion_pipeline import IngestionPipeline
from healthcare_rag.vectordb.chroma_store import ChromaStore
from healthcare_rag.config.settings import get_settings


class IndexManager:
    def __init__(self):
        self.settings=get_settings()
        self.ingestion_pipeline=IngestionPipeline()
        self.vector_store=ChromaStore()

    def build_index(
        self,
        data_dir: Optional[str]=None,
        clear_existing: bool=False
    ) -> None:
        if data_dir is None:
            data_dir=str(self.settings.data_dir/"raw")
        data_path=Path(data_dir)

        if not data_path.exists():
            raise FileNotFoundError(f"data directory not found: {data_dir}")
        print(f"building index from: {data_dir}")
        print(f"Current index size: {self.vector_store.count()} chunks")

        if clear_existing:
            print("clearing existing index...")
            self.vector_store.clear()
        print("Running ingestion pipeline...")
        chunks=self.ingestion_pipeline.ingest(data_dir)

        if not chunks:
            print("No chunks generated. check your directory.")
            return
        self.vector_store.add_chunks(chunks)
        print(f"index build successfully!")
        print(f"Total chunks in index: {self.vector_store.count()}")

    def get_index_stats(self)-> dict:
        return{
            "total-chunks": self.vector_store.count(),
            "collection_name": self.settings.vectordb_collection_name,
            "embedding_model":self.settings.embeddings_model_name,
            "chunk_size":self.settings.rag_chunk_size,
            "chunk_overlap":self.settings.rag_chunk_overlap,
        }        
       