DOC = ("Employees accrue 1.5 days of paid leave per month. Unused leave carries over up to 30 days. "
       "Sick leave is separate and does not accrue.")


def ingest(client, doc_id, text):
    job = client.post("/ingest", json={"doc_id": doc_id, "text": text}).json()["job_id"]
    return client.get(f"/jobs/{job}").json()


def test_roundtrip(client):
    assert ingest(client, "hr-1", DOC)["status"] == "indexed"
    r = client.post("/query", json={"query": "how much leave do I get"}).json()
    assert r["results"][0]["doc_id"] == "hr-1"
    assert r["latency_ms"] < 2000


def test_duplicate_is_skipped(client):
    ingest(client, "a", DOC)
    assert ingest(client, "b", DOC)["status"] == "duplicate"
    assert client.get("/stats").json()["chunks"] == 1


def test_query_embedding_is_cached(client):
    ingest(client, "hr-1", DOC)
    client.post("/query", json={"query": "leave policy"})
    before = client.get("/stats").json()["embed_cache_misses"]
    client.post("/query", json={"query": "leave policy"})
    after = client.get("/stats").json()
    assert after["embed_cache_misses"] == before
    assert after["embed_cache_hits"] >= 1


def test_modes_all_return(client):
    ingest(client, "hr-1", DOC)
    for mode in ("dense", "sparse", "hybrid"):
        r = client.post("/query", json={"query": "sick leave", "mode": mode}).json()
        assert r["results"], mode


def test_ask_without_llm_is_extractive(client):
    ingest(client, "hr-1", DOC)
    r = client.post("/ask", json={"query": "does sick leave accrue"}).json()
    assert r["llm_enabled"] is False
    assert r["citations"][0]["doc_id"] == "hr-1"
    assert "[1]" in r["answer"]


def test_empty_index_ask(client):
    r = client.post("/ask", json={"query": "anything"}).json()
    assert r["citations"] == []


def test_api_key_enforced(client, monkeypatch):
    from app import main
    monkeypatch.setattr(main.settings, "api_key", "secret")
    assert client.get("/stats").status_code == 401
    assert client.get("/stats", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200
