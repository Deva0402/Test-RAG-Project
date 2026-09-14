from typing import List,Dict,Any
from healthcare_rag.rag.retrieval.base_retriever import BaseRetriever
from healthcare_rag.vectordb.chroma_store import ChromaStore

class VectorRetriever(BaseRetriever):
    def __init__(self):
        self.store=ChromaStore()

    def retrieve(
        self,
        query:str,
        top_k: int=5,
        score_threshold: float=0.0,
        **kwargs
    ) -> List[Dict[str,Any]]:
        results=self.store.search(query,top_k=top_k)

        if score_threshold > 0:
            results=[
                r for r in results
                if r.get('score',0) >= score_threshold
            ]
        return results    
          