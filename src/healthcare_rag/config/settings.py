"""Centralized application settings using pydantic Basesettings.
all configuration is loaded from the .env file.
every part of the application imports settings from here"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from pydantic import Field,field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict



class AppEnvironment(str,Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION="production"

class LLMBackend(str,Enum):
    LLAMA_CPP="llama_cpp"
    HUGGINGFACE="huggingface"
    VLLM="vllm"

class EmbeddingBackend(str,Enum):
    SENTENCE_TRANSFORMER="sentence_transformer"
    ONNX="onnx"

class VectorDBBackend(str,Enum):
    CHROMA="chroma"
    FAISS="faiss"
    QDRANT="qdrant"

class AppSettings(BaseSettings):
    model_config=SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    app_name: str =Field(default="Healthcare RAG Assistant")
    app_version: str=Field(default="0.1.0")
    app_env:AppEnvironment = Field(default=AppEnvironment.DEVELOPMENT)
    debug: bool = Field(default=False)
    log_level: str =Field(default="INFO")

    api_host: str= Field(default="0.0.0.0")
    api_port: int= Field(default=8000)

    llm_backend: LLMBackend = Field(default=LLMBackend.LLAMA_CPP)
    llm_model_path: str = Field(default="models/llm/Phi-3-mini-4k-instruct-q4.gguf")
    llm_max_tokens: int= Field(default=512)
    llm_temperature: float=Field(default=0.1)
    llm_n_threads: int=Field(default=4)
    llm_context_length: int = Field(default=4096)
    llm_n_gpu_layers: int = Field(default=0)

    embeddings_model_name: str=Field(default="all-MiniLM-L6-v2")
    embeddings_dimension: int = Field(default=384)
    embeddings_batch_size: int=Field(default=64)

    vectordb_backend: VectorDBBackend=Field(default=VectorDBBackend.CHROMA)
    vectordb_collection_name: str =Field(default="medical_docs")
    chroma_persist_dir: str=Field(default="data/vectorstore/chroma")

    rag_top_k: int=Field(default=5)
    rag_chunk_size: int=Field(default=512)
    rag_chunk_overlap: int=Field(default=64)
    rag_use_hybrid_search: bool =Field(default=True)
    rag_use_reranker: bool =Field(default=True)

    mlflow_tracking_uri: str =Field(default="mlruns")
    mlflow_experiment_name: str =Field(default="healthcare-rag")

    @property
    def base_dir(self)-> Path:
        """Root directory of the project"""
        return Path(__file__).resolve().parents[3]
    
    @property
    def data_dir(self)-> Path:
        return self.base_dir/"data"

    @property
    def models_dir(self)-> Path:
        return self.base_dir/"models"

    @property
    def logs_dir(self)-> Path:
        return self.base_dir/"logs"

    @property
    def is_production(self)-> bool:
        return self.app_env==AppEnvironment.PRODUCTION
    
    @property
    def is_development(self)-> bool:
        return self.app_env==AppEnvironment.DEVELOPMENT

@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Returns a cached singleton Settings instance.
    
    Usage anywhere in the app:
        from healthcare_rag.config.settings import get_settings
        settings=get_settings()
        print(settings.llm_model_path)
        """
    return AppSettings()    

