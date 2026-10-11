import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from heart_disease import config, data, models, validate
from heart_disease.predict import predict_proba


@pytest.fixture(scope="module")
def df():
    return data.load_clean()


def test_load_and_clean(df):
    assert len(df) == 303
    assert set(df[config.TARGET].unique()) == {0, 1}
    assert int(df["ca"].isna().sum()) == 4 and int(df["thal"].isna().sum()) == 2


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_every_model_trains_and_returns_probabilities(df, name):
    X, y = data.features(df), df[config.TARGET]
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    est = models.make_pipeline(name).fit(X_tr, y_tr)
    p = est.predict_proba(X_te)[:, 1]
    assert p.shape == (len(X_te),) and ((p >= 0) & (p <= 1)).all()


def test_imputation_is_inside_the_pipeline(df):
    X = data.features(df).head(40).copy()
    y = df[config.TARGET].head(40)
    X.loc[X.index[0], "ca"] = np.nan
    est = models.make_pipeline("logistic_regression").fit(X, y)
    assert est.predict_proba(X.head(1)).shape == (1, 2)


def test_real_data_passes_validation():
    report = validate.check_training_data(data.load_raw())
    assert report.ok
    warned = {c.name for c in report.warnings}
    assert warned == {"ca has no missing values", "thal has no missing values"}


def test_bad_target_is_an_error():
    raw = data.load_raw().copy()
    raw.loc[0, "num"] = 9
    assert not validate.check_training_data(raw).ok


def test_input_checks_flag_unknown_codes_and_ranges():
    msgs = " ".join(validate.check_inputs({"age": 130, "cp": 0, "thal": 2}))
    assert "Age" in msgs and "Chest pain type" in msgs and "Thallium" in msgs and "Missing" in msgs


def test_predict_proba_from_dicts():
    row = dict(age=58, sex=1, cp=4, trestbps=140, chol=260, fbs=0, restecg=2, thalach=130, exang=1, oldpeak=2.0, slope=2, ca=2, thal=7)
    p = predict_proba([row])[0]
    assert 0.5 < p <= 1.0
