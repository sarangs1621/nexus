import sys, time, pathlib, httpx

BASE = "http://127.0.0.1:8000"
root = pathlib.Path(sys.argv[1])
files = sorted(p for p in root.rglob("*") if p.suffix in {".md", ".txt"})

t0 = time.perf_counter()
jobs = []
with httpx.Client(base_url=BASE, timeout=60) as c:
    for p in files:
        doc_id = str(p.relative_to(root)).replace("\\", "/")
        r = c.post("/ingest", json={"doc_id": doc_id, "text": p.read_text(encoding="utf-8", errors="ignore")})
        jobs.append((doc_id, r.json()["job_id"]))
    pending = dict(jobs)
    while pending:
        for doc_id, job in list(pending.items()):
            s = c.get(f"/jobs/{job}").json()
            if s.get("status") in ("done", "indexed", "duplicate", "failed"):
                pending.pop(doc_id)
                if s.get("status") == "failed":
                    print("FAILED", doc_id, s.get("error"), flush=True)
        time.sleep(0.5)
    stats = c.get("/stats").json()

elapsed = time.perf_counter() - t0
print(f"{len(files)} docs, {stats['chunks']} chunks in {elapsed:.1f}s "
      f"= {len(files)/elapsed:.1f} docs/s = {len(files)/elapsed*86400:,.0f} docs/day", flush=True)
print(f"embed cache: {stats['embed_cache_hits']} hits / {stats['embed_cache_misses']} misses", flush=True)
