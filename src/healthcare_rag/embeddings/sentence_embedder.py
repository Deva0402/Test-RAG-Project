from typing import List
from sentence_transformers import SentenceTransformer
import sentence_transformers
from torch import embedding
from healthcare_rag.embeddings.base_embedder import BaseEmbedder
from healthcare_rag.config.settings import get_settings

class SentenceEmbedder(BaseEmbedder):
    def __init__(self):
        self.settings=get_settings()
        self.model_name=self.settings.embeddings_model_name
        self.batch_size=self.settings.embeddings_batch_size
        print(f"Loading embedding model: {self.model_name}")
        print("This may take a moment on first run(downloading model)...")

        self._model=SentenceTransformer(self.model_name)

        print(f"Embedding model loaded successfully")
    @property
    def dimensions(self)->int:
        return self.settings.embeddings_dimension

    def embed(self,text: str)-> List[float]:
        if not text or not text.strip():
            raise ValueError("cannot embed empty text")

        embedding=self._model.encode(
            text,
            normalize_embeddings=True
        ) 
        return embedding.tolist()
    def embed_batch(self,texts: List[str])->  List[List[float]]:
        if not texts:
            return[]
        valid_texts=[t for t in texts if t and t.strip()]

        if not valid_texts:

            return[]
        print(f"Embedding{len(valid_texts)} texts in batches of {self.batch_size}...")
        embeddings=self._model.encode(
            valid_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )   
        return embeddings.tolist()