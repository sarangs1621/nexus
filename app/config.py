from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # infra
    redis_url: str = "redis://localhost:6379/0"
    use_fake_redis: bool = True
    index_dir: str = "data/index"
    api_key: str | None = None            # if set, every route except /health needs X-API-Key

    # models
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    anthropic_api_key: str | None = None  # if unset, /ask falls back to extractive answers
    llm_model: str = "claude-sonnet-5-5"

    # retrieval
    chunk_size: int = 300
    chunk_overlap: int = 50
    top_k: int = 5
    candidate_k: int = 10
    rerank_default: bool = False
    rrf_dense_weight: float = 1.0
    rrf_sparse_weight: float = 1.0

    model_config = {"env_prefix": "NEXUS_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
