from pathlib import Path

from fastapi.testclient import TestClient

from app.api import ModelService, app
from app.config import FEATURE_COLUMNS

VALID_PAYLOAD = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


def test_prediction_and_health_endpoints() -> None:
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert set(response.json()) == {"prediction", "confidence", "model_version"}
    assert response.json()["prediction"] in (0, 1)
    assert 0 <= response.json()["confidence"] <= 1


def test_invalid_payload_is_rejected() -> None:
    client = TestClient(app)
    invalid = {key: 0 for key in FEATURE_COLUMNS}
    invalid["age"] = 200
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_metrics_change_after_request() -> None:
    client = TestClient(app)
    before = client.get("/metrics").text
    client.post("/predict", json=VALID_PAYLOAD)
    after = client.get("/metrics").text
    assert "heart_disease_api_requests_total" in after
    assert after != before


def test_missing_model_is_not_ready(tmp_path: Path) -> None:
    missing = ModelService(tmp_path / "missing.joblib")
    assert missing.ready is False
