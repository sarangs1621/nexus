from app.chunking import chunk_text
from app.retrieval import rrf


def test_chunk_overlap():
    text = " ".join(str(i) for i in range(10))
    chunks = chunk_text(text, size=4, overlap=1)
    assert chunks[0].split() == ["0", "1", "2", "3"]
    assert chunks[1].split()[0] == "3"
    assert chunks[-1].split()[-1] == "9"


def test_chunk_empty():
    assert chunk_text("   ", 4, 1) == []


def test_rrf_prefers_items_in_both_lists():
    fused = rrf([[1, 2, 3], [3, 4, 5]])
    assert max(fused, key=fused.get) == 3