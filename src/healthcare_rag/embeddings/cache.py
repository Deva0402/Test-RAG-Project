import hashlib
import json
from typing import List,Optional
from pathlib import Path
import diskcache
from healthcare_rag.config.settings import get_settings

class EmbeddingCache:
    def __init__(self):
        self.settings=get_settings()
        cache_dir=Path(self.settings.embeddings_model_name.replace("/","_"))
        cache_path=Path("data/embeddings/cache")/cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache=diskcache.Cache(str(cache_path))
        print(f"embedding cache initialized at: {cache_path}")

    def _make_key(self,text: str, model_name: str) ->str:
        content=f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    def get(self,text:str)-> Optional[List[float]]:
        key=self._make_key(text,self.settings.embeddings_model_name)
        return self.cache.get(key)
    def set(self, text: str, embedding: List[float])->None:
        key=self._make_key(text,self.settings.embeddings_model_name)
        self.cache.set(key,embedding)
    def get_batch(self, texts:List[str])-> dict:
        hits={}
        misses={}

        for text in texts:
            cached=self.get(text)
            if cached is not None:
                hits[text]=cached
            else:
                misses.append(text)
        return {"hits":hits,"misses":misses}
    def set_batch(self,texts: List[str],embeddings: List[List[float]])->None:
        for text, embbedding in zip(texts, embeddings):
            self.set(text, embeddings)
    @property
    def size(self)->int:
        return len(self.cache)
    def clear(self)->None:
        self.cache.clear()
        print("embedding cache cleared")                           
