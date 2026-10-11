"""Predict heart disease risk for one patient with the saved model.

Usage:
    python -m heart_disease.predict --json '{"age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0,
        "restecg": 2, "thalach": 130, "exang": 1, "oldpeak": 2.0, "slope": 2, "ca": 2, "thal": 7}'
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

from . import config
from .validate import check_inputs


def load_model(path=config.MODEL_PATH):
    if not path.exists():
        from . import train
        train.main()
    return joblib.load(path)


def predict_proba(rows: list[dict], model=None) -> np.ndarray:
    model = model or load_model()
    X = pd.DataFrame(rows).reindex(columns=config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)
    for c in config.CATEGORICAL_FEATURES:
        X[c] = X[c].astype("Int64")
    return model.predict_proba(X)[:, 1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", required=True, help="One patient as a JSON object")
    row = json.loads(p.parse_args().json)
    for msg in check_inputs(row):
        print(f"warning: {msg}", file=sys.stderr)
    print(f"Estimated probability of heart disease: {predict_proba([row])[0]:.1%}")


if __name__ == "__main__":
    main()
