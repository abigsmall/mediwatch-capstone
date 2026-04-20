from mediwatch.api.app import app


class DummyModel:
    def predict_proba(self, frame):
        return [[0.2, 0.8]]


def test_health():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr("mediwatch.api.app.load_model", lambda: DummyModel())
    monkeypatch.setattr(
        "mediwatch.api.app.read_manifest",
        lambda: {"model_version": "unit-test"},
    )

    client = TestClient(app)
    response = client.post("/predict", json={"age": "[70-80)", "time_in_hospital": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == "readmitted"
    assert payload["model_version"] == "unit-test"
