"""Figures for the README (written to reports/figures)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from . import config

RED, BLUE, GREY, INK = "#D1495B", "#2E6F95", "#B8C4CC", "#1F2A33"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlesize": 11.5, "axes.edgecolor": "#9AA7B0", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def target_and_age(df: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 2]})
    counts = df[config.TARGET].value_counts().reindex([0, 1])
    ax[0].bar(["No disease", "Disease"], counts.values, color=[BLUE, RED])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v + 3, f"{v} ({v / len(df):.0%})", ha="center", fontsize=9.5)
    ax[0].set_title("Patients by diagnosis"); ax[0].set_ylim(0, counts.max() * 1.2)
    bins = [28, 40, 50, 60, 70, 80]
    g = df.groupby(pd.cut(df["age"], bins, labels=["29-40", "41-50", "51-60", "61-70", "71-77"]), observed=True)[config.TARGET]
    rate, n = g.mean(), g.size()
    ax[1].bar(rate.index.astype(str), rate.values * 100, color=RED)
    for i, (r, k) in enumerate(zip(rate.values, n.values)):
        ax[1].text(i, r * 100 + 2, f"{r:.0%}\n(n={k})", ha="center", fontsize=9)
    ax[1].set_title("Share with heart disease by age"); ax[1].set_ylabel("% with disease"); ax[1].set_ylim(0, 100)
    _save(fig, "01_target_and_age.png")


def key_features(df: pd.DataFrame):
    feats = [("cp", "Chest pain type"), ("thal", "Thallium stress test"), ("exang", "Exercise-induced angina"), ("ca", "Major vessels (fluoroscopy)")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for ax, (c, title) in zip(axes, feats):
        g = df.dropna(subset=[c]).groupby(c)[config.TARGET]
        rate = g.mean()
        labels = [config.CODES.get(c, {}).get(int(k), str(int(k))) for k in rate.index]
        ax.bar(range(len(rate)), rate.values * 100, color=[RED if r >= 0.5 else BLUE for r in rate.values])
        ax.set_xticks(range(len(rate)), [l.replace(" ", "\n", 1) for l in labels], fontsize=8.5)
        for i, (r, k) in enumerate(zip(rate.values, g.size().values)):
            ax.text(i, r * 100 + 2, f"{r:.0%}", ha="center", fontsize=9)
        ax.set_title(title); ax.set_ylim(0, 105)
    axes[0].set_ylabel("% with disease")
    _save(fig, "02_disease_rate_by_feature.png")


def model_comparison(rows: pd.DataFrame, best: str):
    rows = rows.sort_values("cv_auc_mean")
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    y = np.arange(len(rows)); h = 0.38
    ax.barh(y + h / 2, rows["cv_auc_mean"], height=h, xerr=rows["cv_auc_std"], capsize=3,
            color=[RED if m == best else BLUE for m in rows.index], label="Cross-validated (mean ± sd, 50 folds)")
    ax.barh(y - h / 2, rows["test_roc_auc"], height=h, color=GREY, label="Single 61-patient test set")
    for i, (m, sd, t) in enumerate(zip(rows["cv_auc_mean"], rows["cv_auc_std"], rows["test_roc_auc"])):
        ax.text(m + sd + 0.006, i + h / 2, f"{m:.3f}", va="center", fontsize=9, fontweight="bold")
        ax.text(t + 0.006, i - h / 2, f"{t:.3f}", va="center", fontsize=8.5, color="#5B6B7A")
    ax.set_yticks(y, rows.index); ax.set_xlim(0.7, 1.03); ax.set_xlabel("ROC AUC")
    ax.set_title("Model comparison: the single test split flatters every model")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5, frameon=False)
    _save(fig, "03_model_comparison.png")


def roc_curves(fitted: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    palette = [RED, BLUE, "#E8A33D", "#3C8D5A"]
    for (name, est), c in zip(fitted.items(), palette):
        p = est.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the 61 test patients"); ax.legend(fontsize=8, frameon=False, loc="lower right")
    _save(fig, "04_roc_curves.png")


def confusion(y_test, proba, name):
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=15, fontweight="bold", color="white" if v > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["No disease", "Disease"]); ax.set_yticks([0, 1], ["No disease", "Disease"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{name}: test confusion matrix")
    _save(fig, "05_confusion_matrix.png")


def importance(imp: pd.Series):
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([config.LABELS[c] for c in imp.index], imp.values, color=[RED if v >= imp.max() * 0.5 else BLUE for v in imp.values])
    ax.set_xlabel("Drop in ROC AUC when the feature is shuffled")
    ax.set_title("What drives the prediction (permutation importance, test set)")
    _save(fig, "06_feature_importance.png")


def pca_projection(prep, X_train, y_train):
    Z = prep.fit_transform(X_train)
    p = PCA(n_components=2, random_state=config.RANDOM_STATE).fit(Z)
    P = p.transform(Z)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for v, c, lab in [(0, BLUE, "No disease"), (1, RED, "Disease")]:
        m = np.asarray(y_train) == v
        ax.scatter(P[m, 0], P[m, 1], s=18, alpha=.7, color=c, label=lab)
    ev = p.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    ax.set_title("PCA of the training patients"); ax.legend(frameon=False)
    _save(fig, "07_pca.png")
