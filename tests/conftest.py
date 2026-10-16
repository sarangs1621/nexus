import os, tempfile
os.environ.setdefault("NEXUS_USE_FAKE_REDIS", "true")
os.environ.setdefault("NEXUS_INDEX_DIR", tempfile.mkdtemp())
os.environ.pop("NEXUS_ANTHROPIC_API_KEY", None)
os.environ.pop("NEXUS_API_KEY", None)

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clean_index(client):
    client.delete("/admin/index")
    client.delete("/admin/cache")
