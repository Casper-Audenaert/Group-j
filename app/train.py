import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "v0.1"
SEED = 42
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


def build_model():
    return make_pipeline(StandardScaler(), LinearRegression())


def train(out_dir="model"):
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X[FEATURES], y, test_size=0.2, random_state=SEED)
    model = build_model().fit(X_tr, y_tr)
    pred = model.predict(X_te)

    threshold = float(np.percentile(y_tr, 75))
    actual, flagged = y_te >= threshold, pred >= threshold
    metrics = {
        "model_version": MODEL_VERSION,
        "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred))), 4),
        "high_risk_threshold": round(threshold, 2),
        "precision_at_threshold": round(float(precision_score(actual, flagged)), 4),
        "recall_at_threshold": round(float(recall_score(actual, flagged)), 4),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "seed": SEED,
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "version": MODEL_VERSION, "features": FEATURES}, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train(*sys.argv[1:])
