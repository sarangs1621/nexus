import os
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from .embeddings import Embedder
from .index import HybridIndex


def rrf(rankings: list[list[int]], weights: list[float] | None = None, k: int = 60) -> dict[int, float]:
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + w / (k + rank + 1)
    return fused


class Retriever:
    def __init__(self, index: HybridIndex, embedder: Embedder, rerank_model: str,
                 dense_weight: float = 1.0, sparse_weight: float = 1.0):
        self.index, self.embedder = index, embedder
        self.weights = {"dense": dense_weight, "sparse": sparse_weight}
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self.reranker = CrossEncoder(rerank_model, max_length=256)
        # warm-up so the first real request doesn't pay model-init cost
        self.reranker.predict([("warm", "up")], show_progress_bar=False)

    def candidates(self, query: str, candidate_k: int, mode: str) -> list[int]:
        rankings, weights = [], []
        if mode in ("dense", "hybrid"):
            qvec = self.embedder.embed([query])[0]
            rankings.append([c for c, _ in self.index.dense(qvec, candidate_k)])
            weights.append(self.weights["dense"])
        if mode in ("sparse", "hybrid"):
            rankings.append([c for c, _ in self.index.sparse(query, candidate_k)])
            weights.append(self.weights["sparse"])
        fused = rrf(rankings, weights)
        return sorted(fused, key=fused.get, reverse=True)[:candidate_k]

    def rerank(self, query: str, cands: list[int]) -> list[tuple[int, float]]:
        pairs = [(query, self.index.meta[c]["text"]) for c in cands]
        scores = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        return sorted(zip(cands, map(float, scores)), key=lambda x: x[1], reverse=True)

    def search(self, query: str, top_k: int, candidate_k: int,
               rerank: bool = False, mode: str = "hybrid",
               extra_queries: list[str] | None = None) -> list[dict]:
        """Retrieve for `query` (plus optional rewritten variants), fuse, optionally rerank."""
        per_query = [self.candidates(q, candidate_k, mode) for q in [query, *(extra_queries or [])]]
        fused = rrf(per_query)
        cands = sorted(fused, key=fused.get, reverse=True)[:candidate_k]
        if not cands:
            return []
        scored = self.rerank(query, cands) if rerank else [(c, fused[c]) for c in cands]
        return [{"chunk_id": c, "score": s, **self.index.meta[c]} for c, s in scored[:top_k]]