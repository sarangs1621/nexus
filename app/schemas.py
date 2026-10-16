from typing import Literal
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    doc_id: str
    text: str = Field(min_length=1)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = None
    rerank: bool | None = None
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"


class Chunk(BaseModel):
    chunk_id: int
    doc_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    results: list[Chunk]
    latency_ms: float


class AskRequest(QueryRequest):
    rewrite: bool = True


class Citation(BaseModel):
    n: int
    chunk_id: int
    doc_id: str


class AskResponse(BaseModel):
    query: str
    rewritten_query: str | None
    answer: str
    citations: list[Citation]
    passages: list[Chunk]
    llm_enabled: bool
    latency_ms: float