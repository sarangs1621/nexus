import logging
import time
import uuid
from contextlib import asynccontextmanager

import redis
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool

from .config import settings
from .embeddings import Embedder
from .index import HybridIndex
from .ingest import Ingestor
from .llm import AnthropicLLM, Generator
from .retrieval import Retriever
from .schemas import AskRequest, AskResponse, Citation, IngestRequest, QueryRequest, QueryResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("nexus")


class State:
    r: redis.Redis
    embedder: Embedder
    index: HybridIndex
    ingestor: Ingestor
    retriever: Retriever
    generator: Generator


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.use_fake_redis:
        import fakeredis
        state.r = fakeredis.FakeRedis()
        log.warning("using fakeredis (in-memory); set NEXUS_USE_FAKE_REDIS=false for real Redis")
    else:
        state.r = redis.from_url(settings.redis_url)
        state.r.ping()
    state.embedder = Embedder(settings.embed_model, state.r)
    state.index = HybridIndex(state.embedder.dim, settings.index_dir)
    state.ingestor = Ingestor(state.index, state.embedder, state.r, settings.chunk_size, settings.chunk_overlap)
    state.retriever = Retriever(state.index, state.embedder, settings.rerank_model,
                                settings.rrf_dense_weight, settings.rrf_sparse_weight)
    llm = AnthropicLLM(settings.anthropic_api_key, settings.llm_model) if settings.anthropic_api_key else None
    state.generator = Generator(llm)
    log.info("ready: %d chunks indexed, llm=%s", state.index.size, state.generator.enabled)
    yield


async def require_key(x_api_key: str | None = Header(default=None)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(401, "invalid or missing X-API-Key")


app = FastAPI(title="Nexus", version="1.0.0", lifespan=lifespan)
protected = [Depends(require_key)]


# ---------------- ingestion ----------------
def _run_ingest(job_id: str, doc_id: str, text: str):
    key = f"job:{job_id}"
    state.r.hset(key, "status", "running")
    try:
        result = state.ingestor.ingest(doc_id, text)
        state.r.hset(key, mapping={"status": "done", **{k: str(v) for k, v in result.items()}})
        log.info("ingest %s -> %s (%s chunks)", doc_id, result["status"], result["chunks"])
    except Exception as e:  # noqa: BLE001
        log.exception("ingest failed for %s", doc_id)
        state.r.hset(key, mapping={"status": "failed", "error": str(e)})


@app.post("/ingest", status_code=202, dependencies=protected)
async def ingest(req: IngestRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex
    state.r.hset(f"job:{job_id}", mapping={"status": "queued", "doc_id": req.doc_id})
    bg.add_task(_run_ingest, job_id, req.doc_id, req.text)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}", dependencies=protected)
async def job(job_id: str):
    data = state.r.hgetall(f"job:{job_id}")
    if not data:
        raise HTTPException(404, "job not found")
    return {k.decode(): v.decode() for k, v in data.items()}


# ---------------- retrieval ----------------
@app.post("/query", response_model=QueryResponse, dependencies=protected)
async def query(req: QueryRequest):
    t0 = time.perf_counter()
    rerank = settings.rerank_default if req.rerank is None else req.rerank
    results = await run_in_threadpool(
        state.retriever.search, req.query, req.top_k or settings.top_k,
        settings.candidate_k, rerank, req.mode,
    )
    return QueryResponse(query=req.query, results=results, latency_ms=(time.perf_counter() - t0) * 1000)


@app.post("/ask", response_model=AskResponse, dependencies=protected)
async def ask(req: AskRequest):
    t0 = time.perf_counter()
    rewritten = await run_in_threadpool(state.generator.rewrite, req.query) if req.rewrite else None
    rerank = settings.rerank_default if req.rerank is None else req.rerank
    passages = await run_in_threadpool(
        state.retriever.search, req.query, req.top_k or settings.top_k,
        settings.candidate_k, rerank, req.mode, [rewritten] if rewritten else None,
    )
    answer, cited = await run_in_threadpool(state.generator.answer, req.query, passages)
    citations = [Citation(n=n, chunk_id=passages[n - 1]["chunk_id"], doc_id=passages[n - 1]["doc_id"]) for n in cited]
    return AskResponse(
        query=req.query, rewritten_query=rewritten, answer=answer, citations=citations,
        passages=passages, llm_enabled=state.generator.enabled,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


# ---------------- ops ----------------
@app.get("/health")
async def health():
    return {"ok": True, "chunks": state.index.size, "llm": state.generator.enabled}


@app.get("/stats", dependencies=protected)
async def stats():
    return {"chunks": state.index.size, "docs": len({m["doc_id"] for m in state.index.meta.values()}),
            "embed_cache_hits": state.embedder.hits, "embed_cache_misses": state.embedder.misses}


@app.delete("/admin/cache", dependencies=protected)
async def clear_embed_cache():
    keys = list(state.r.scan_iter("emb:*"))
    if keys:
        state.r.delete(*keys)
    state.embedder.hits = state.embedder.misses = 0
    return {"deleted": len(keys)}


@app.delete("/admin/index", dependencies=protected)
async def clear_index():
    state.index.clear()
    for k in state.r.scan_iter("doc:*"):
        state.r.delete(k)
    return {"ok": True}