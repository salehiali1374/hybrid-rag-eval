from fastapi.testclient import TestClient

from hybrid_rag.api.app import app


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
