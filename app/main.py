import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parent.parent / "model" / "model.joblib"))
bundle = joblib.load(MODEL_PATH)

app = FastAPI(title="Diabetes progression risk", version=bundle["version"])

Feature = Field(allow_inf_nan=False)


class Patient(BaseModel):
    age: float = Feature
    sex: float = Feature
    bmi: float = Feature
    bp: float = Feature
    s1: float = Feature
    s2: float = Feature
    s3: float = Feature
    s4: float = Feature
    s5: float = Feature
    s6: float = Feature


@app.get("/health")
def health():
    return {"status": "ok", "model_version": bundle["version"]}


@app.post("/predict")
def predict(patient: Patient):
    row = pd.DataFrame([patient.model_dump()])[bundle["features"]]
    return {"prediction": float(bundle["model"].predict(row)[0])}
