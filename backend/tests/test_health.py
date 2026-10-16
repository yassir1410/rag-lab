from fastapi.testclient import TestClient


def test_health_liveness(client: TestClient):
    """Liveness probe must return status ok with version and environment."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data


def test_health_readiness(client: TestClient):
    """Readiness probe verifies database and storage connectivity."""
    response = client.get("/api/v1/health/ready")
    assert response.status_code in (200, 503)
    data = response.json()
    assert "database_connected" in data
    assert "storage_writable" in data
