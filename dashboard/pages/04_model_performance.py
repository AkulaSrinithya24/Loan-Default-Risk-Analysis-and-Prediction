"""
dashboard/pages/04_model_performance.py
-----------------------------------------
Model Performance page — metrics, ROC, PR curve, feature importance.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import (
    RAW_DATA_PATH, RANDOM_STATE, TEST_SIZE, DECISION_THRESHOLD,
    MODEL_PATH, DEPLOYMENT_MODEL_PATH,
)

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

from dashboard.theme import page_header, section_title, show_table

page_header("Model Performance",
            "ROC/PR curves, confusion matrix, feature importances and threshold analysis")

# ── Resolve which model file to use ────────────────────────────────────────────
# Priority: full model (local) → deployment model (cloud) → helpful error
if MODEL_PATH.exists():
    _model_path = MODEL_PATH
elif DEPLOYMENT_MODEL_PATH.exists():
    _model_path = DEPLOYMENT_MODEL_PATH
    st.info(
        "Running with the **deployment model** (30 trees).  \n"
        "Metrics will be slightly different from the full 200-tree model "
        "used in the academic report.  \n"
        "To use the full model locally, run `python run_pipeline.py`."
    )
else:
    st.warning(
        "No model file found. Generate the deployment model first:\n\n"
        "```\npython scripts/create_deployment_model.py\n```\n\n"
        "Or run the full training pipeline:\n\n"
        "```\npython run_pipeline.py\n```"
    )
    st.stop()

# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    import joblib
    from src.data_loader import load_data
    from src.data_cleaner import clean_data
    from src.feature_engineering import build_preprocessor
    from sklearn.model_selection import train_test_split

    pipeline = joblib.load(_model_path)
    df = clean_data(load_data(RAW_DATA_PATH))
    X, y, _ = build_preprocessor(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return pipeline, X_test, y_test

pipeline, X_test, y_test = load_model_and_data()

# ── Metrics ────────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report,
)

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

section_title("Model Metrics Summary")
st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC",        f"{roc_auc_score(y_test, y_prob):.4f}")
c2.metric("Avg Precision",  f"{average_precision_score(y_test, y_prob):.4f}")
c3.metric("F1 (Default)",   f"{f1_score(y_test, y_pred):.4f}")
c4.metric("Accuracy",       f"{accuracy_score(y_test, y_pred):.4f}")

st.markdown(
    f"<p style='color:#5a6a80;font-size:0.82rem;margin:0.5rem 0 0 0;'>"
    f"Decision threshold: <strong style='color:#0a2342;'>{DECISION_THRESHOLD}</strong></p>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Row 1: Confusion Matrix  |  Classification Report ─────────────────────────
row1_left, row1_right = st.columns([1, 1])

with row1_left:
    section_title("Confusion Matrix")
    st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)
    import seaborn as sns
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor="white")
    sns.heatmap(
        cm, annot=True, fmt=",", cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        linewidths=0.5, linecolor="white", ax=ax,
    )
    ax.set_title(f"Confusion Matrix (threshold={DECISION_THRESHOLD})",
                 fontweight="bold", color="#0a2342", fontsize=10)
    ax.set_xlabel("Predicted", color="#5a6a80", fontsize=8)
    ax.set_ylabel("Actual", color="#5a6a80", fontsize=8)
    ax.tick_params(labelcolor="#5a6a80", labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with row1_right:
    section_title("Classification Report")
    st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)
    cr = classification_report(
        y_test, y_pred, target_names=["No Default", "Default"], output_dict=True
    )
    cr_df = pd.DataFrame(cr).T.round(4)
    show_table(cr_df)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Row 2: ROC Curve  |  Precision-Recall Curve ────────────────────────────────
row2_left, row2_right = st.columns([1, 1])

with row2_left:
    section_title("ROC Curve")
    st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor="white")
    ax2.set_facecolor("white")
    ax2.plot(fpr, tpr, color="#1a56a0", linewidth=2.5,
             label=f"Random Forest (AUC={auc:.4f})")
    ax2.plot([0, 1], [0, 1], color="#dce3ed", linewidth=1.2,
             linestyle="--", label="Random Baseline")
    ax2.fill_between(fpr, tpr, alpha=0.06, color="#1a56a0")
    ax2.set_xlabel("False Positive Rate", color="#5a6a80", fontsize=8)
    ax2.set_ylabel("True Positive Rate", color="#5a6a80", fontsize=8)
    ax2.set_title("ROC Curve", fontweight="bold", color="#0a2342", fontsize=10)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.tick_params(labelcolor="#5a6a80", labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color("#dce3ed")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

with row2_right:
    section_title("Precision-Recall Curve")
    st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    fig3, ax3 = plt.subplots(figsize=(5, 4), facecolor="white")
    ax3.set_facecolor("white")
    ax3.plot(rec, prec, color="#b91c1c", linewidth=2.5, label=f"AP={ap:.4f}")
    ax3.axhline(y_test.mean(), color="#dce3ed", linestyle="--", linewidth=1.2,
                label=f"Baseline ({y_test.mean():.3f})")
    ax3.fill_between(rec, prec, alpha=0.06, color="#b91c1c")
    ax3.set_xlabel("Recall", color="#5a6a80", fontsize=8)
    ax3.set_ylabel("Precision", color="#5a6a80", fontsize=8)
    ax3.set_title("Precision-Recall Curve", fontweight="bold", color="#0a2342", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.tick_params(labelcolor="#5a6a80", labelsize=8)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.spines[["left", "bottom"]].set_color("#dce3ed")
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Row 3: Feature Importances  |  Threshold Analysis ─────────────────────────
row3_left, row3_right = st.columns([1, 1])

with row3_left:
    section_title("Feature Importances")
    st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)
    from src.feature_engineering import get_feature_names
    clf  = pipeline.named_steps["classifier"]
    prep = pipeline.named_steps["preprocessor"]
    names = get_feature_names(prep)
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig4, ax4 = plt.subplots(figsize=(5, 4.5), facecolor="white")
    ax4.set_facecolor("white")
    top = 12
    top_idx = idx[:top]
    ax4.barh(
        [names[i] for i in top_idx][::-1],
        importances[top_idx][::-1],
        color="#1a56a0", edgecolor="white", height=0.6,
    )
    ax4.set_xlabel("Feature Importance (Gini)", color="#5a6a80", fontsize=8)
    ax4.set_title(f"Top {top} Feature Importances — Random Forest",
                  fontweight="bold", color="#0a2342", fontsize=10)
    ax4.tick_params(labelcolor="#5a6a80", labelsize=8)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.spines[["left", "bottom"]].set_color("#dce3ed")
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

with row3_right:
    section_title("Threshold Analysis")
    st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)
    from sklearn.metrics import precision_score, recall_score
    thresholds = np.arange(0.10, 0.80, 0.05)
    rows = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        rows.append({
            "Threshold": round(float(t), 2),
            "Precision": round(precision_score(y_test, yp, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, yp, zero_division=0), 4),
            "F1":        round(f1_score(y_test, yp, zero_division=0), 4),
        })
    thresh_df = pd.DataFrame(rows)

    fig5, ax5 = plt.subplots(figsize=(5, 4.5), facecolor="white")
    ax5.set_facecolor("white")
    ax5.plot(thresh_df["Threshold"], thresh_df["Precision"], label="Precision",
             color="#1a56a0", marker="o", markersize=4, linewidth=2)
    ax5.plot(thresh_df["Threshold"], thresh_df["Recall"], label="Recall",
             color="#b91c1c", marker="o", markersize=4, linewidth=2)
    ax5.plot(thresh_df["Threshold"], thresh_df["F1"], label="F1",
             color="#1a7a4a", marker="o", markersize=4, linewidth=2)
    ax5.axvline(DECISION_THRESHOLD, color="#0a2342", linestyle="--", linewidth=1.5,
                label=f"Chosen ({DECISION_THRESHOLD})")
    ax5.set_xlabel("Decision Threshold", color="#5a6a80", fontsize=8)
    ax5.set_ylabel("Score", color="#5a6a80", fontsize=8)
    ax5.set_title("Precision / Recall / F1 vs Threshold",
                  fontweight="bold", color="#0a2342", fontsize=10)
    ax5.legend(fontsize=8)
    ax5.tick_params(labelcolor="#5a6a80", labelsize=8)
    ax5.spines[["top", "right"]].set_visible(False)
    ax5.spines[["left", "bottom"]].set_color("#dce3ed")
    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Feature importance table (full, below the 2-col layout) ───────────────────
section_title("Full Feature Importance Table")
st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)
fi_df = pd.DataFrame({"Feature": names, "Importance": importances})\
          .sort_values("Importance", ascending=False).reset_index(drop=True)
show_table(fi_df, fmt={"Importance": "{:.6f}"})
