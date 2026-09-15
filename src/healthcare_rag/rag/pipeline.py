from typing import List,Dict, Any,Optional

from httpx import get
from healthcare_rag.rag.retrieval.base_retriever import BaseRetriever
from healthcare_rag.rag.retrieval.vector_retriever import VectorRetriever
from healthcare_rag.rag.retrieval.hybrid_retriever import HybridRetriever
from healthcare_rag.models.llama_cpp_model import LlamaCppModel
from healthcare_rag.rag.prompts.system_prompts import MEDICAL_SYSTEM_PROMPT,MEDICAL_QA_PROMPT

class RAGPipeline:
    def __init__(
        self,
        retriever: Optional[BaseRetriever]=None,
        retriever_type: str="vector",
        
        documents: Optional[List[Dict[str, Any]]]=None    
    ):
        if retriever:
            self.retriever=retriever
        elif retriever_type=="hybrid" and documents:
            self.retriever=HybridRetriever(documents=documents)
        else:
            self.retriever=VectorRetriever()

        self.llm=LlamaCppModel()

        self.system_prompt=MEDICAL_SYSTEM_PROMPT

    def _build_context(self,retrieved_docs: List[Dict[str,Any]]) -> str :

        context_parts: List[str]=[]

        for i, doc in enumerate(retrieved_docs,1):
            text=doc.get("content") or doc.get("text") or ""
            metadata=doc.get('metadata',{})
            source=(
                doc.get("source") or metadata.get("filename") or "unknown"
            )

            context_parts.append(f"[source{i}:{source}]\n{text}")
        return "\n\n".join(context_parts)

    def query(
        self,
        question: str,
        top_k: int=5,
        max_tokens: int=512,
        temperature: float=0.1
    ) -> Dict[str,Any]:
        if not question or not question.strip():
            raise ValueError("question cannot be empty")
        
        retrieved_docs=self.retriever.retrieve(question,top_k=top_k)

        if not retrieved_docs:
            return {
                "answer":("I couldn't find relevant information in the medical database to answer your question.please consult with a healthcare professional."),
                "sources":[],
                "retrieved_count":0,
                "context":"",
            }
        context=self._build_context(retrieved_docs)

        prompt=MEDICAL_QA_PROMPT.format(
            context=context,
            question=question
        )

        messages=[
            {"role":"system","content":self.system_prompt},
            {"role":"user","content":prompt},
        ]

        answer=self.llm.generate_chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        sources: List[str]=[]
            
        for doc in retrieved_docs:
            metadata=doc.get('metadata',{})
            src=doc.get("source") or metadata.get("source") or metadata.get("filename")
            if src:
                sources.append(str(src))
            
        
        return {
            "answer":answer,
            "source": list(set(sources)),
            "retrieved_count": len(retrieved_docs),
            "context": context
        }          