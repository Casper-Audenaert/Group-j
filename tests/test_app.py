import importlib
import os
import tempfile

from fastapi.testclient import TestClient

from app import train

MODEL_DIR = tempfile.mkdtemp()
METRICS = train.train(MODEL_DIR)
os.environ["MODEL_PATH"] = os.path.join(MODEL_DIR, "model.joblib")

app = importlib.import_module("app.main").app

client = TestClient(app)
PATIENT = {"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02,
           "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": train.MODEL_VERSION}


def test_predict_returns_float():
    r = client.post("/predict", json=PATIENT)
    assert r.status_code == 200
    assert list(r.json()) == ["prediction"]
    assert isinstance(r.json()["prediction"], float)


def test_missing_feature_is_422_naming_field():
    body = {k: v for k, v in PATIENT.items() if k != "s6"}
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "s6"]


def test_wrong_type_is_422():
    r = client.post("/predict", json={**PATIENT, "bmi": "high"})
    assert r.status_code == 422


def test_training_is_deterministic():
    assert train.train(tempfile.mkdtemp()) == METRICS
