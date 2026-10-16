import hashlib
import numpy as np
import redis
from sentence_transformers import SentenceTransformer


class Embedder:
    """Sentence embeddings with a Redis cache keyed by chunk hash."""

    def __init__(self, model_name: str, r: redis.Redis):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()
        self.r = r
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(text: str) -> str:
        return "emb:" + hashlib.sha256(text.encode()).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        cached = self.r.mget([self._key(t) for t in texts])
        missing = []
        for i, blob in enumerate(cached):
            if blob:
                out[i] = np.frombuffer(blob, dtype=np.float32)
                self.hits += 1
            else:
                missing.append(i)
                self.misses += 1
        if missing:
            vecs = self.model.encode(
                [texts[i] for i in missing],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            pipe = self.r.pipeline()
            for i, v in zip(missing, vecs):
                out[i] = v
                pipe.set(self._key(texts[i]), v.tobytes())
            pipe.execute()
        return np.vstack(out)