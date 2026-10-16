import hashlib
import redis
from .chunking import chunk_text
from .embeddings import Embedder
from .index import HybridIndex


class Ingestor:
    def __init__(self, index: HybridIndex, embedder: Embedder, r: redis.Redis, size: int, overlap: int):
        self.index, self.embedder, self.r = index, embedder, r
        self.size, self.overlap = size, overlap

    def ingest(self, doc_id: str, text: str) -> dict:
        digest = hashlib.sha256(text.encode()).hexdigest()
        # NX set = atomic dedup; same content under any doc_id is skipped
        if not self.r.set(f"doc:{digest}", doc_id, nx=True):
            return {"doc_id": doc_id, "status": "duplicate", "chunks": 0}
        chunks = chunk_text(text, self.size, self.overlap)
        if not chunks:
            return {"doc_id": doc_id, "status": "empty", "chunks": 0}
        vecs = self.embedder.embed(chunks)
        self.index.add(doc_id, chunks, vecs)
        return {"doc_id": doc_id, "status": "indexed", "chunks": len(chunks)}