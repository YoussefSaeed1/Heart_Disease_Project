"""Model zoo. Every model is a full scikit-learn pipeline (imputation, scaling, one-hot encoding,
classifier), so cross-validation never leaks information from the validation folds."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from . import config


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric, config.NUMERIC_FEATURES),
                              ("cat", categorical, config.CATEGORICAL_FEATURES)])


def make_pipeline(name: str) -> Pipeline:
    rs = config.RANDOM_STATE
    clf = {
        "baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(C=0.3, max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=3, random_state=rs, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=rs),
        "svm": CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", random_state=rs), ensemble=False, cv=5),
    }[name]
    return Pipeline([("prep", preprocessor()), ("model", clf)])


MODEL_NAMES = ["baseline", "logistic_regression", "random_forest", "gradient_boosting", "svm"]
PRETTY = {"baseline": "Baseline (always 'no disease' rate)", "logistic_regression": "Logistic Regression",
          "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting", "svm": "SVM (RBF)"}
