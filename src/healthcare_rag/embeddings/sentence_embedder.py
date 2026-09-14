from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from healthcare_rag.config.settings import get_settings
from healthcare_rag.embeddings.base_embedder import BaseEmbedder
from healthcare_rag.embeddings.cache import EmbeddingCache


class SentenceEmbedder(BaseEmbedder):

    def __init__(self):
        self.settings = get_settings()

        self.model_name = (
            self.settings.embeddings_model_name
        )

        self.batch_size = (
            self.settings.embeddings_batch_size
        )

        self._model = SentenceTransformer(
            self.model_name
        )

        self.cache = EmbeddingCache()

    @property
    def dimensions(self) -> int:
        dimensions = self._model.get_sentence_embedding_dimension()

        if dimensions is None:
            raise RuntimeError(
                "Could not determine embedding dimensions"
            )

        return dimensions

    def embed(self, text: str) -> List[float]:

        if not text or not text.strip():
            raise ValueError(
                "Cannot embed empty text"
            )

        cached = self.cache.get(text)

        if cached is not None:
            return cached

        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        result = [float(value) for value in embedding]

        self.cache.set(
            text,
            result,
        )

        return result

    def embed_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        if not texts:
            return []

        results: List[List[float]] = [
            [] for _ in texts
        ]

        missing_texts = []
        missing_indices = []

        for index, text in enumerate(texts):

            if not text or not text.strip():
                raise ValueError(
                    f"Text at index {index} is empty"
                )

            cached = self.cache.get(text)

            if cached is not None:
                results[index] = cached
            else:
                missing_texts.append(text)
                missing_indices.append(index)

        if missing_texts:

            embeddings = self._model.encode(
                missing_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

            for index, text, embedding in zip(
                missing_indices,
                missing_texts,
                embeddings,
            ):
                result = [float(value) for value in embedding]

                results[index] = result

                self.cache.set(
                    text,
                    result,
                )

        return results