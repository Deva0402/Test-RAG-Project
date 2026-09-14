from typing import List,Dict,Any
import math
from collections import Counter
from healthcare_rag.rag.retrieval.base_retriever import BaseRetriever

class BM25Retriever(BaseRetriever):
    def __init__(self,documents:List[Dict[str,Any]]):
        self.documents=documents
        self.doc_freqs=[self._tokenize(doc.get('text',''))for doc in documents]
        self.idf=self._compute_idf()
        self.avg_doc_len=sum(len(df) for df in self.doc_freqs)/len(documents) if documents else 0
        self.k1=1.5
        self.b=0.75

    def _tokenize(self,text: str)-> Counter:
        tokens=text.lower().split()
        return Counter(tokens)

    def _compute_idf(self) -> Dict[str,float]:
        idf={}
        n_docs=len(self.doc_freqs)

        for doc_freq in self.doc_freqs:
            for term in doc_freq:
                idf[term]=idf.get(term,0)+1
        for term in idf:
            idf[term]=math.log((n_docs-idf[term]+0.5)/(idf[term]+0.5)+1)
        return idf
    def retrieve(
        self,
        query:str,
        top_k:int=5,
        **kwargs
    ) -> List[Dict[str , Any]]:
        query_tokens=self._tokenize(query)
        scores=[]

        for i,doc_freq in enumerate(self.doc_freqs):
            score=0
            doc_len=len(doc_freq)

            for term,q_freq in query_tokens.items():
                if term in doc_freq:
                    d_freq=doc_freq[term]
                    idf=self.idf.get(term,0)
                    numerator=d_freq* (self.k1+1)
                    denominator=d_freq+self.k1*(1-self.b+ self.b* (doc_len/self.avg_doc_len))
                    score+=idf*(numerator/denominator)
            scores.append((i,score))
        scores.sort(key=lambda x:x[1],reverse=True)
        results=[]

        for idx, score in scores[:top_k]:
            doc=self.documents[idx].copy()
            doc['score']=score
            results.append(doc)
        return results        

                    