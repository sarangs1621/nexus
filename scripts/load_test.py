import asyncio, json, pathlib, statistics, sys, time, httpx

BASE = "http://127.0.0.1:8000"


async def worker(c, i, lat, qs, rounds):
    for r in range(rounds):
        q = qs[(i + r) % len(qs)]
        t0 = time.perf_counter()
        resp = await c.post("/query", json={"query": q, "mode": "hybrid"})
        resp.raise_for_status()
        lat.append((time.perf_counter() - t0) * 1000)


async def main():
    concurrency = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    qs = [json.loads(l)["query"] for l in pathlib.Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    lat = []
    async with httpx.AsyncClient(base_url=BASE, timeout=60) as c:
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(c, i, lat, qs, rounds) for i in range(concurrency)))
        wall = time.perf_counter() - t0
    lat.sort()
    print(f"concurrency={concurrency} requests={len(lat)} wall={wall:.1f}s rps={len(lat)/wall:.1f}", flush=True)
    print(f"p50={lat[len(lat)//2]:.0f}ms p95={lat[int(len(lat)*.95)]:.0f}ms p99={lat[int(len(lat)*.99)]:.0f}ms "
          f"mean={statistics.mean(lat):.0f}ms", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
