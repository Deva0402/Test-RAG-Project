import hashlib
from pathlib import Path
from typing import Dict, List, Optional,cast,TypedDict

import diskcache

from healthcare_rag.config.settings import get_settings

class BatchCacheResult(TypedDict):
    hits: Dict[str, List[float]]
    misses: List[str]

class EmbeddingCache:

    def __init__(self):
        self.settings = get_settings()

        cache_dir = Path(
            self.settings.embeddings_model_name.replace("/", "_")
        )

        cache_path = (
            self.settings.base_dir
            / "data"
            / "embeddings"
            / "cache"
            / cache_dir
        )

        cache_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.cache = diskcache.Cache(
            str(cache_path)
        )

    def _make_key(
        self,
        text: str,
        model_name: str,
    ) -> str:

        content = f"{model_name}:{text}"

        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    def get(
        self,
        text: str,
    ) -> Optional[List[float]]:

        key = self._make_key(
            text,
            self.settings.embeddings_model_name,
        )

        value = self.cache.get(key)

        if value is None:
            return None
        return cast(List[float],value)

    def set(
        self,
        text: str,
        embedding: List[float],
    ) -> None:

        key = self._make_key(
            text,
            self.settings.embeddings_model_name,
        )

        self.cache.set(
            key,
            embedding,
        )

    def get_batch(
    self,
    texts: List[str],
    ) -> BatchCacheResult:

        hits = {}
        misses = []

        for text in texts:

            cached = self.get(text)

            if cached is not None:
                hits[text] = cached
            else:
                misses.append(text)

        return {
            "hits": hits,
            "misses": misses,
        }

    def set_batch(
        self,
        texts: List[str],
        embeddings: List[List[float]],
    ) -> None:

        for text, embedding in zip(
            texts,
            embeddings,
        ):
            self.set(
                text,
                embedding,
            )

    @property
    def size(self) -> int:
        return cast(int, self.cache.stats(enable=True)[0])

    def clear(self) -> None:
        self.cache.clear()