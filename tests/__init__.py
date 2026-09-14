"""Healthcare RAG Assistant
A production-grade Retreival-Augmented Generation system 
for medical question answering.

Tech Stack:
-LLM: llama.cpp(CPU-optimized, quantized models)
-Embeddings: sentence-transformers(all-MIniLM-L6-v2)
-vector DB: ChromaDB(local,persistent)
-RAG: LangChain
-API : FastAPI
-MLOps: MLflow + Prometheus
"""

__version__ = "0.1.0"
__author__ = "Devasish"
