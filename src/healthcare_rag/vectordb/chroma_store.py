from typing import List, Dict, Any, Optional,cast
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.api.types import Embedding
from healthcare_rag.embeddings.sentence_embedder import SentenceEmbedder
from healthcare_rag.data.processors.chunker import Chunk
from healthcare_rag.config.settings import get_settings


class ChromaStore:

    def __init__(self):
        self.settings = get_settings()
        self.embedder = SentenceEmbedder()

        persist_dir = Path(self.settings.chroma_persist_dir)

        # Convert relative path to project-root-relative path
        if not persist_dir.is_absolute():
            persist_dir = self.settings.base_dir / persist_dir

        persist_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=self.settings.vectordb_collection_name,
            metadata={
                "hnsw:space": "cosine"
            }
        )

        print(
            f"ChromaDB initialized | "
            f"collection: {self.settings.vectordb_collection_name}"
        )

        print(
            f"Current documents in store: "
            f"{self.collection.count()}"
        )

    def add_chunks(
        self,
        chunks: List[Chunk]
    ) -> None:

        if not chunks:
            print("No chunks to add")
            return

        print(
            f"Adding {len(chunks)} chunks to ChromaDB..."
        )

        # -----------------------------------------
        # 1. Extract text
        # -----------------------------------------

        texts = [
            chunk.content
            for chunk in chunks
        ]

        # -----------------------------------------
        # 2. Generate embeddings
        # -----------------------------------------

        embeddings = self.embedder.embed_batch(texts)

        # -----------------------------------------
        # 3. Generate UNIQUE IDs
        # -----------------------------------------

        ids = []

        for chunk in chunks:

            document_index = chunk.metadata.get(
                "index",
                0
            )

            chunk_id = (
                f"{chunk.source}_"
                f"{document_index}_"
                f"{chunk.chunk_index}"
            )

            ids.append(chunk_id)

        # -----------------------------------------
        # 4. Prepare metadata
        # -----------------------------------------

        metadatas = []

        for chunk in chunks:

            metadata = {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }

            for key, value in chunk.metadata.items():

                if isinstance(
                    value,
                    (str, int, float, bool)
                ):
                    metadata[key] = value

            metadatas.append(metadata)

        # -----------------------------------------
        # 5. Upsert in batches
        # -----------------------------------------

        batch_size = 100

        for i in range(
            0,
            len(chunks),
            batch_size
        ):

            batch_ids = ids[i:i + batch_size]

            batch_embeddings = cast(
                List[Embedding],
                embeddings[i:i + batch_size]
            )

            batch_texts = texts[
                i:i + batch_size
            ]

            batch_metadatas = metadatas[
                i:i + batch_size
            ]

            self.collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas
            )

        print(
            f"Successfully added "
            f"{len(chunks)} chunks to ChromaDB"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[
            Dict[str, Any]
        ] = None
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        collection_count = self.collection.count()

        if collection_count == 0:
            return []

        query_embedding = self.embedder.embed(
            query
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(
                top_k,
                collection_count
            ),
            where=filter_metadata,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        formatted_results = []

        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")

        if not documents or not metadatas or not distances:
            return formatted_results

        for doc, metadata, distance in zip(
            documents[0],
            metadatas[0],
            distances[0]
        ):

            formatted_results.append({
                "content": doc,
                "metadata": metadata,
                "score": 1 - distance,
                "source": metadata.get(
                    "source",
                    "unknown"
                )
            })

        return formatted_results

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:

        self.client.delete_collection(
            self.settings.vectordb_collection_name
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.settings.vectordb_collection_name,
                metadata={
                    "hnsw:space": "cosine"
                }
            )
        )

        print(
            "ChromaDB collection cleared"
        )