import json
import os
import threading
import faiss
import numpy as np
from rank_bm25 import BM25Okapi


import re

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class HybridIndex:
    """FAISS (dense, inner product on normalized vectors) + BM25 (sparse), persisted to disk."""

    def __init__(self, dim: int, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.lock = threading.Lock()
        self.meta: dict[int, dict] = {}
        self.ids: list[int] = []
        self.next_id = 0
        self.faiss = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.bm25: BM25Okapi | None = None
        self._load()

    # --- persistence -------------------------------------------------
    def _files(self):
        return os.path.join(self.path, "faiss.index"), os.path.join(self.path, "meta.json")

    def _load(self):
        f, m = self._files()
        if os.path.exists(f) and os.path.exists(m):
            self.faiss = faiss.read_index(f)
            with open(m) as fh:
                self.meta = {int(k): v for k, v in json.load(fh).items()}
            self.next_id = (max(self.meta) + 1) if self.meta else 0
            self._rebuild_bm25()

    def save(self):
        f, m = self._files()
        faiss.write_index(self.faiss, f)
        with open(m, "w") as fh:
            json.dump(self.meta, fh)

    def _rebuild_bm25(self):
        self.ids = sorted(self.meta)
        corpus = [tokenize(self.meta[i]["text"]) for i in self.ids]
        if corpus:
            self.bm25 = BM25Okapi(corpus)
            for w, val in self.bm25.idf.items():
                if val <= 0:
                    self.bm25.idf[w] = 0.01
        else:
            self.bm25 = None

    # --- writes --------------------------------------------------------
    def add(self, doc_id: str, chunks: list[str], vecs: np.ndarray):
        with self.lock:
            ids = np.arange(self.next_id, self.next_id + len(chunks), dtype=np.int64)
            self.faiss.add_with_ids(vecs, ids)
            for cid, text in zip(ids, chunks):
                self.meta[int(cid)] = {"doc_id": doc_id, "text": text}
            self.next_id += len(chunks)
            self._rebuild_bm25()  # rank_bm25 is immutable; fine at this scale
            self.save()

    def clear(self):
        with self.lock:
            self.faiss.reset()
            self.meta, self.ids, self.next_id, self.bm25 = {}, [], 0, None
            self.save()

    # --- reads ---------------------------------------------------------
    def dense(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.faiss.ntotal == 0:
            return []
        scores, ids = self.faiss.search(qvec.reshape(1, -1).astype(np.float32), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(self.ids[j], float(scores[j])) for j in top if scores[j] > 0]

    @property
    def size(self) -> int:
        return self.faiss.ntotal