from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = "ollama-fastapi-robot-proxy"

    ollama_local_base_url: str = Field(default="http://localhost:11434")
    ollama_remote_urls: str = Field(default="http://127.0.0.1:11434")

    remote_primary_model: str = Field(default="qwen2.5:7b-instruct")
    remote_secondary_model: str = Field(default="llama3.1:8b-instruct")
    local_fallback_model: str = Field(default="qwen2.5:1.5b-instruct")

    code_model: str = Field(default="deepseek-coder:6.7b")
    embedding_model: str = Field(default="nomic-embed-text")
    vision_model: str = Field(default="llava:7b")

    request_timeout_seconds: float = Field(default=45.0, ge=1.0)
    health_timeout_seconds: float = Field(default=3.0, ge=0.5)
    max_retries: int = Field(default=1, ge=0, le=5)

    knowledge_base_dir: str = Field(default="./knowledge_base")
    chroma_persist_dir: str = Field(default="./chroma_store")
    chroma_collection_name: str = Field(default="company_kb")
    rag_chunk_size: int = Field(default=800, ge=100)
    rag_chunk_overlap: int = Field(default=120, ge=0)
    rag_top_k: int = Field(default=4, ge=1, le=10)
    rag_candidate_pool_multiplier: int = Field(default=4, ge=1, le=10)

    enable_query_rewrite: bool = Field(default=True)
    rag_distance_threshold: float = Field(default=0.46, ge=0.0, le=2.0)
    rag_dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    rag_keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    rag_min_fusion_score: float = Field(default=0.05, ge=0.0, le=1.0)
    rag_context_max_chars: int = Field(default=2800, ge=500, le=12000)
    rag_context_max_chunks: int = Field(default=4, ge=1, le=10)
    rag_dedupe_similarity_threshold: float = Field(default=0.92, ge=0.5, le=1.0)
    rag_log_path: str = Field(default="./logs/rag_events.jsonl")
    memory_max_turns: int = Field(default=8, ge=1, le=50)

    # Keeps a model resident in (V)RAM between requests instead of reloading it every call.
    # "30m" = stay loaded for 30 minutes of inactivity; "-1" = stay loaded indefinitely.
    ollama_keep_alive: str = Field(default="30m")

    def remote_urls(self) -> List[str]:
        return [url.strip().rstrip("/") for url in self.ollama_remote_urls.split(",") if url.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
