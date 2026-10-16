import json, statistics, pathlib, httpx

BASE = "http://127.0.0.1:8000"
qs = [json.loads(l) for l in pathlib.Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
MODES = [("dense", False), ("sparse", False), ("hybrid", False), ("hybrid", True)]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


rows = []
with httpx.Client(base_url=BASE, timeout=120) as c:
    for mode, rerank in MODES:
        c.delete("/admin/cache")  # every mode starts with a cold query-embedding cache
        h1 = h5 = 0; rr, lat = [], []
        for q in qs:
            r = c.post("/query", json={"query": q["query"], "top_k": 5, "mode": mode, "rerank": rerank}).json()
            lat.append(r["latency_ms"])
            seen = []
            for x in r["results"]:
                if x["doc_id"] not in seen:
                    seen.append(x["doc_id"])
            if seen and seen[0] == q["doc_id"]: h1 += 1
            if q["doc_id"] in seen:
                h5 += 1; rr.append(1 / (seen.index(q["doc_id"]) + 1))
            else:
                rr.append(0.0)
        n = len(qs)
        rows.append((mode + ("+rerank" if rerank else ""), h1 / n, h5 / n, statistics.mean(rr), pct(lat, .5), pct(lat, .95)))

print(f"\n{len(qs)} questions, cold query-embedding cache per mode\n", flush=True)
print("| mode | hit@1 | hit@5 | MRR | p50 ms | p95 ms |", flush=True)
print("|---|---|---|---|---|---|", flush=True)
for name, a, b, m, p50, p95 in rows:
    print(f"| {name} | {a:.2f} | {b:.2f} | {m:.2f} | {p50:.0f} | {p95:.0f} |", flush=True)
