"""Validate the data, compare models with repeated cross-validation, test the best one, save it.

Usage:
    python -m heart_disease.train
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split

from . import config, data, evaluate, models, plots, validate


def main() -> None:
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    raw = data.load_raw()
    report = validate.assert_valid(raw)
    with open(config.REPORTS_DIR / "data_validation.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Data validation passed ({len(report.warnings)} warnings)")

    df = data.clean(raw)
    X, y = data.features(df), df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=config.CV_REPEATS, random_state=config.RANDOM_STATE)

    rows, fitted = {}, {}
    for name in models.MODEL_NAMES:
        est = models.make_pipeline(name)
        scores = cross_val_score(est, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        est.fit(X_tr, y_tr)
        test = evaluate.report(y_te, est.predict_proba(X_te)[:, 1])
        rows[models.PRETTY[name]] = {"cv_auc_mean": scores.mean(), "cv_auc_std": scores.std(),
                                     **{f"test_{k}": v for k, v in test.items() if k != "confusion"}}
        fitted[name] = (est, test)
        print(f"{models.PRETTY[name]:<38} CV AUC {scores.mean():.3f} ± {scores.std():.3f}   test AUC {test['roc_auc']:.3f}  recall {test['recall']:.3f}")

    table = pd.DataFrame(rows).T
    table.to_csv(config.REPORTS_DIR / "model_comparison.csv", float_format="%.4f")
    candidates = table.drop(index=models.PRETTY["baseline"])
    best_pretty = candidates["cv_auc_mean"].idxmax()
    best = {v: k for k, v in models.PRETTY.items()}[best_pretty]
    best_est, best_test = fitted[best]
    print(f"\nBest model by cross-validated AUC: {best_pretty}")

    imp = permutation_importance(best_est, X_te, y_te, scoring="roc_auc", n_repeats=50, random_state=config.RANDOM_STATE)
    importance = pd.Series(imp.importances_mean, index=X_te.columns)

    summary = {
        "patients": len(df), "disease_share": float(y.mean()), "train_rows": len(X_tr), "test_rows": len(X_te),
        "cv": f"{config.CV_FOLDS}-fold stratified, repeated {config.CV_REPEATS} times, training set only",
        "best_model": best_pretty,
        "models": {k: {m: float(v) for m, v in r.items()} for k, r in rows.items()},
        "best_test_confusion": best_test["confusion"],
        "permutation_importance": {k: float(v) for k, v in importance.sort_values(ascending=False).items()},
    }
    with open(config.REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # figures
    plots.target_and_age(df)
    plots.key_features(df)
    plots.model_comparison(table.drop(index=models.PRETTY["baseline"]).rename(index=lambda s: s.replace(" (RBF)", "")), best_pretty.replace(" (RBF)", ""))
    plots.roc_curves({models.PRETTY[n].replace(" (RBF)", ""): fitted[n][0] for n in models.MODEL_NAMES if n != "baseline"}, X_te, y_te)
    plots.confusion(y_te, best_est.predict_proba(X_te)[:, 1], best_pretty)
    plots.importance(importance)
    plots.pca_projection(models.preprocessor(), X_tr, y_tr)

    final = models.make_pipeline(best)
    final.fit(X, y)
    joblib.dump(final, config.MODEL_PATH)
    print(f"Saved model -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
