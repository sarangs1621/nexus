# Nexus — Hybrid Semantic Search & RAG

FastAPI service that ingests documents, indexes them with FAISS (dense) + BM25 (sparse),
fuses results with reciprocal rank fusion, optionally reranks with a cross-encoder,
and answers questions with cited passages via an LLM.

## Architecture

```
POST /ingest ──▶ dedup (sha256, Redis NX) ──▶ chunk ──▶ embed (Redis-cached) ──▶ FAISS + BM25 (persisted)
POST /query  ──▶ dense ┐
                sparse ┴─▶ RRF ──▶ [cross-encoder rerank] ──▶ top-k
POST /ask    ──▶ query rewrite (LLM) ──▶ /query for both ──▶ grounded answer with [n] citations
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # optional: add NEXUS_ANTHROPIC_API_KEY
uvicorn app.main:app --reload                        # Swagger at http://127.0.0.1:8000/docs
```
Without an Anthropic key, `/ask` returns an extractive answer (top passage) so the full pipeline still runs.
With Docker: `docker compose up --build`.

## Evaluation

Corpus: FastAPI documentation, 156 markdown files → 728 chunks. 40 hand-labelled questions
(`eval/questions.jsonl`), each tagged with the document that answers it. Reproduce with
`python scripts/ingest_dir.py <path>` then `python eval/run_eval.py`.

| mode | hit@1 | hit@5 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|
| dense | 0.62 | 0.95 | 0.76 | 25 | 123 |
| sparse | 0.55 | 0.85 | 0.68 | 3 | 5 |
| hybrid | 0.68 | 0.85 | 0.76 | 29 | 100 |
| hybrid+rerank | 0.55 | 0.85 | 0.69 | 455 | 585 |

Latencies are per-request on a laptop CPU (AMD Ryzen 7 5800H), cold query-embedding cache.

**Findings.** Hybrid retrieval achieved the highest Hit@1 accuracy (0.68) with an MRR of 0.76 at 29 ms p50, successfully fusing exact keyword hits from BM25 with semantic recall from dense embeddings. Dense alone excelled at broad recall (Hit@5 of 0.95), while sparse alone was extremely fast (3 ms p50) but dropped accuracy on paraphrased queries. Cross-encoder reranking on CPU added significant latency (~455 ms p50) without improving accuracy over RRF for this corpus, confirming why reranking is disabled by default (`NEXUS_RERANK_DEFAULT=false`) for CPU-bound deployments.

## Load

`python scripts/load_test.py 20 5` — 147.5 RPS, p50=115ms, p95=165ms at 20 concurrent clients.

Ingestion: 156 docs in 52.5s (3.0 docs/s = ~256,779 docs/day) on the same machine, single process.

## Endpoints

| route | purpose |
|---|---|
| `POST /ingest` | queue a document; returns `job_id` |
| `GET /jobs/{id}` | job status: queued / running / done / duplicate / failed |
| `POST /query` | retrieve chunks; `mode` = dense / sparse / hybrid, `rerank` bool |
| `POST /ask` | rewrite → retrieve → answer with citations |
| `GET /health`, `GET /stats` | liveness; chunk/doc counts and cache hit rate |
| `DELETE /admin/cache`, `DELETE /admin/index` | reset embedding cache / index |

Set `NEXUS_API_KEY` to require `X-API-Key` on everything except `/health`.

## Tests

`pytest -q` — ingest/query round trip, dedup, cache hits, all three modes, `/ask` fallback, auth.
CI runs tests and a Docker build on every push.

## Known limits

- BM25 index is rebuilt on every ingest (rank_bm25 is immutable); fine to ~10⁵ chunks, after that swap for a persistent inverted index.
- Single-process; FAISS `IndexFlatIP` is exact search — switch to IVF/HNSW past ~10⁶ vectors.
- fakeredis loses cache/dedup state on restart; use real Redis for anything persistent.
