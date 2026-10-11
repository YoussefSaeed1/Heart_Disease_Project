"""Loading the UCI Cleveland file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_raw(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the comma-separated file; '?' marks a missing value."""
    return pd.read_csv(path, header=None, names=config.COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Binary target (any disease) and integer codes. Missing values stay NaN: they are
    imputed inside the model pipeline, so imputation is learned from training rows only."""
    out = df.copy()
    out[config.TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns=["num"])
    for c in config.CATEGORICAL_FEATURES:
        out[c] = out[c].astype("Int64")
    return out


def load_clean(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


def features(df: pd.DataFrame) -> pd.DataFrame:
    return df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
