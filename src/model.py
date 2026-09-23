
"""
model.py
--------
Training, evaluation, persistence and inference for the loan default
prediction model.

Public API
----------
train_model(X_train, y_train, preprocessor)   -> Pipeline
evaluate_model(pipeline, X_test, y_test)       -> dict
save_artefacts(pipeline, preprocessor)         -> None
load_pipeline()                                -> Pipeline
predict_default(pipeline, input_df)            -> dict
plot_confusion_matrix(pipeline, X_test, y_test) -> Figure
plot_roc_curve(pipeline, X_test, y_test)        -> Figure
plot_pr_curve(pipeline, X_test, y_test)         -> Figure
plot_feature_importance(pipeline, feature_names) -> Figure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score,
)
from sklearn.compose import ColumnTransformer

from src.config import (
    RF_PARAMS,
    RANDOM_STATE,
    DECISION_THRESHOLD,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURE_NAMES_PATH,
    MODELS_DIR,
)
from src.feature_engineering import get_feature_names


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """Fit a full sklearn Pipeline: preprocessor + RandomForestClassifier.

    Parameters
    ----------
    X_train      : raw feature DataFrame (NOT yet transformed)
    y_train      : target Series
    preprocessor : unfitted ColumnTransformer from build_preprocessor()

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    clf = RandomForestClassifier(**RF_PARAMS)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])
    pipeline.fit(X_train, y_train)
    print(f"[train_model] Training complete. "
          f"Trees: {RF_PARAMS['n_estimators']}, "
          f"OOB available: {hasattr(clf, 'oob_score_')}")
    return pipeline


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """Fit a Logistic Regression baseline pipeline."""
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   lr),
    ])
    pipeline.fit(X_train, y_train)
    print("[train_baseline] Logistic Regression baseline trained.")
    return pipeline


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DECISION_THRESHOLD,
) -> dict:
    """Evaluate the pipeline and print a full metrics report.

    Uses a custom probability threshold for the final label decision
    to handle class imbalance better than the default 0.5.

    Returns
    -------
    dict with keys: accuracy, roc_auc, avg_precision, f1,
                    classification_report, confusion_matrix
    """
    y_prob  = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_prob >= threshold).astype(int)

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_pr  = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)
    cm      = confusion_matrix(y_test, y_pred)
    cr      = classification_report(y_test, y_pred,
                                    target_names=["No Default", "Default"])

    print("=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)
    print(f"  Threshold     : {threshold}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  ROC-AUC       : {roc_auc:.4f}")
    print(f"  Avg Precision : {avg_pr:.4f}")
    print(f"  F1 (Default)  : {f1:.4f}")
    print()
    print("  Classification Report:")
    print(cr)
    print("  Confusion Matrix:")
    print(cm)
    print("=" * 60)

    return {
        "accuracy":               round(acc,     4),
        "roc_auc":                round(roc_auc, 4),
        "avg_precision":          round(avg_pr,  4),
        "f1":                     round(f1,      4),
        "classification_report":  cr,
        "confusion_matrix":       cm,
    }


# ── Persistence ────────────────────────────────────────────────────────────────

def save_artefacts(pipeline: Pipeline, preprocessor: ColumnTransformer) -> None:
    """Save the fitted pipeline, preprocessor and feature names to models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline,    MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    feature_names = get_feature_names(preprocessor)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    print(f"[save_artefacts] Pipeline saved  -> {MODEL_PATH}")
    print(f"[save_artefacts] Preprocessor    -> {PREPROCESSOR_PATH}")
    print(f"[save_artefacts] Feature names   -> {FEATURE_NAMES_PATH}")


def load_pipeline() -> Pipeline:
    """Load and return the saved pipeline from models/."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No saved model found at {MODEL_PATH}. "
            "Run `python run_pipeline.py` first."
        )
    return joblib.load(MODEL_PATH)


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_default(
    pipeline: Pipeline,
    input_df: pd.DataFrame,
    threshold: float = DECISION_THRESHOLD,
) -> dict:
    """Run inference on a single-row or multi-row DataFrame.

    Parameters
    ----------
    pipeline  : fitted Pipeline from load_pipeline()
    input_df  : DataFrame with the same feature columns as training data
    threshold : decision threshold for labelling

    Returns
    -------
    dict with keys: probability, prediction, risk_label
    """
    prob  = pipeline.predict_proba(input_df)[:, 1]
    label = (prob >= threshold).astype(int)
    risk  = ["Very High" if p >= 0.6 else
             "High"      if p >= 0.4 else
             "Medium"    if p >= 0.2 else
             "Low"       for p in prob]
    return {
        "probability": prob.tolist(),
        "prediction":  label.tolist(),
        "risk_label":  risk,
    }


# ── Evaluation plots ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DECISION_THRESHOLD,
) -> plt.Figure:
    """Annotated confusion matrix heatmap."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns_cm = __import__("seaborn")
    sns_cm.heatmap(
        cm, annot=True, fmt=",", cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        linewidths=0.5, linecolor="white",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix (threshold={threshold})", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_roc_curve(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> plt.Figure:
    """ROC curve with AUC annotation."""
    y_prob  = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4a90d9", linewidth=2,
            label=f"Random Forest (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Loan Default Prediction", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr_curve(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> plt.Figure:
    """Precision-Recall curve with Average Precision annotation."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#e05c5c", linewidth=2,
            label=f"Random Forest (AP = {ap:.4f})")
    ax.axhline(baseline, color="k", linestyle="--", linewidth=1,
               alpha=0.5, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Loan Default Prediction",
                 fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_feature_importance(
    pipeline: Pipeline,
    top_n: int = 16,
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances from the Random Forest."""
    clf   = pipeline.named_steps["classifier"]
    prep  = pipeline.named_steps["preprocessor"]
    names = get_feature_names(prep)

    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    top_names  = [names[i] for i in idx]
    top_values = importances[idx]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
    bars = ax.barh(
        top_names[::-1], top_values[::-1],
        color="#4a90d9", edgecolor="white", linewidth=0.6,
    )
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest",
                 fontweight="bold")
    fig.tight_layout()
    return fig
