from typing import List,Dict,Any,Optional
from healthcare_rag.rag.retrieval.base_retriever import BaseRetriever
from healthcare_rag.rag.retrieval.vector_retriever import VectorRetriever
from healthcare_rag.rag.retrieval.bm25_retriever import BM25Retriever

class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        collection_name: str="medical_docs",
        documents: Optional[List[Dict[str,Any]]]=None,
        k: int =60
    ):  
        self.vector_retriever=VectorRetriever()
        self.bm25_retriever=BM25Retriever(documents) if documents else None
        self.k=k
    def retrieve(
        self,
        query: str,
        top_k: int=5,
        **kwargs
    ) -> List[Dict[str,Any]]:

        vector_results=self.vector_retriever.retrieve(query,top_k=top_k*2)

        if self.bm25_retriever:
            bm25_results=self.bm25_retriever.retrieve(query,top_k=top_k*2)
        else:
            bm25_results=[]

        rrf_scores={}  

        for rank,result in enumerate(vector_results,1):
            doc_id=result.get('id',str(result))
            rrf_scores[doc_id]={
                'doc':result,
                'score':1/(self.k+rank)
            }  

        
        for rank,result in enumerate(bm25_results,1):
            doc_id=result.get('id',str(result))
            if doc_id in rrf_scores:
                rrf_scores[doc_id]['score']+=1/(self.k+rank)
            else:
                rrf_scores[doc_id]={
                    'doc':result,
                    'score':1/(self.k+rank)
                }           
        results=[
            {**data['doc'],'score':data['score']}
            for data in rrf_scores.values()
        ]
        results.sort(key=lambda x:x['score'],reverse=True)
        return results[ :top_k]            