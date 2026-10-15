import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import api.main as api  # noqa: E402
from src.forecast import ModelNotFoundError  # noqa: E402


@pytest.fixture()
def client(trained_env):
    api._CACHE.clear()
    return TestClient(api.app)


def test_health_and_locations(client):
    assert client.get("/health").json()["model_loaded"] is True
    assert set(client.get("/locations").json()["locations"]) == {"TestCityA", "TestCityB"}
    assert "model" in client.get("/model-info").json()


def test_forecast_ok(client):
    r = client.post("/forecast", json={"location": "TestCityA", "horizon": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 5 and body["unit"] == "µg/m³"
    assert all(x["pm25_pred"] >= 0 for x in body["forecast"])


@pytest.mark.parametrize("payload", [{"location": "TestCityA", "horizon": 0}, {"location": "TestCityA", "horizon": 8},
                                     {"location": ""}, {"horizon": 3}])
def test_validation_errors(client, payload):
    assert client.post("/forecast", json=payload).status_code == 422


def test_unsupported_city(client):
    assert client.post("/forecast", json={"location": "Atlantis"}).status_code == 404


def test_alerts_endpoint(client):
    r = client.post("/alerts", json={"location": "TestCityA", "threshold": 1})
    assert r.status_code == 200 and r.json()["alerts"]


def test_missing_model_gives_503(monkeypatch):
    def boom():
        raise ModelNotFoundError("no model")

    monkeypatch.setattr(api, "get_context", boom)
    c = TestClient(api.app)
    assert c.post("/forecast", json={"location": "X"}).status_code == 503
    assert c.get("/health").json()["status"] == "degraded"
