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

from src.config import RAW_DATA_PATH, RANDOM_STATE, TEST_SIZE, DECISION_THRESHOLD, MODEL_PATH

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")
st.title("🤖 Model Performance")
st.markdown("---")

# ── Check model exists ─────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.warning(
        "No trained model found. Please run the training pipeline first:\n\n"
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

    pipeline = joblib.load(MODEL_PATH)
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

st.subheader("Model Metrics Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC",        f"{roc_auc_score(y_test, y_prob):.4f}")
c2.metric("Avg Precision",  f"{average_precision_score(y_test, y_prob):.4f}")
c3.metric("F1 (Default)",   f"{f1_score(y_test, y_pred):.4f}")
c4.metric("Accuracy",       f"{accuracy_score(y_test, y_pred):.4f}")

st.markdown(f"*Decision threshold: **{DECISION_THRESHOLD}***")
st.markdown("---")

# ── Confusion matrix ───────────────────────────────────────────────────────────
st.subheader("Confusion Matrix")
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt=",", cmap="Blues",
    xticklabels=["No Default", "Default"],
    yticklabels=["No Default", "Default"],
    linewidths=0.5, linecolor="white", ax=ax,
)
ax.set_title(f"Confusion Matrix (threshold={DECISION_THRESHOLD})", fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
st.pyplot(fig)
plt.close(fig)

st.markdown("---")

# ── Classification report ──────────────────────────────────────────────────────
st.subheader("Classification Report")
cr = classification_report(
    y_test, y_pred, target_names=["No Default", "Default"], output_dict=True
)
cr_df = pd.DataFrame(cr).T.round(4)
st.dataframe(cr_df, use_container_width=True)

st.markdown("---")

# ── ROC curve ─────────────────────────────────────────────────────────────────
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(fpr, tpr, color="#4a90d9", linewidth=2, label=f"Random Forest (AUC={auc:.4f})")
ax2.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Baseline")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve", fontweight="bold")
ax2.legend(loc="lower right")
st.pyplot(fig2)
plt.close(fig2)

st.markdown("---")

# ── Precision-Recall curve ─────────────────────────────────────────────────────
st.subheader("Precision-Recall Curve")
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

fig3, ax3 = plt.subplots(figsize=(6, 5))
ax3.plot(rec, prec, color="#e05c5c", linewidth=2, label=f"AP={ap:.4f}")
ax3.axhline(y_test.mean(), color="k", linestyle="--", linewidth=1, alpha=0.5,
            label=f"Baseline ({y_test.mean():.3f})")
ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
ax3.set_title("Precision-Recall Curve", fontweight="bold")
ax3.legend(loc="upper right")
st.pyplot(fig3)
plt.close(fig3)

st.markdown("---")

# ── Feature importance ─────────────────────────────────────────────────────────
st.subheader("Feature Importances")
from src.feature_engineering import get_feature_names
clf  = pipeline.named_steps["classifier"]
prep = pipeline.named_steps["preprocessor"]
names = get_feature_names(prep)
importances = clf.feature_importances_
idx = np.argsort(importances)[::-1]

fig4, ax4 = plt.subplots(figsize=(9, 5))
top = 16
top_idx = idx[:top]
ax4.barh(
    [names[i] for i in top_idx][::-1],
    importances[top_idx][::-1],
    color="#4a90d9", edgecolor="white",
)
ax4.set_xlabel("Feature Importance (Gini)")
ax4.set_title(f"Top {top} Feature Importances — Random Forest", fontweight="bold")
st.pyplot(fig4)
plt.close(fig4)

fi_df = pd.DataFrame({"Feature": names, "Importance": importances})\
          .sort_values("Importance", ascending=False).reset_index(drop=True)
st.dataframe(fi_df.style.format({"Importance": "{:.6f}"}), use_container_width=True)

st.markdown("---")

# ── Threshold analysis ─────────────────────────────────────────────────────────
st.subheader("Threshold Analysis")
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

fig5, ax5 = plt.subplots(figsize=(9, 4))
ax5.plot(thresh_df["Threshold"], thresh_df["Precision"], label="Precision",
         color="#4a90d9", marker="o", markersize=4)
ax5.plot(thresh_df["Threshold"], thresh_df["Recall"], label="Recall",
         color="#e05c5c", marker="o", markersize=4)
ax5.plot(thresh_df["Threshold"], thresh_df["F1"], label="F1",
         color="#27ae60", marker="o", markersize=4)
ax5.axvline(DECISION_THRESHOLD, color="k", linestyle="--", linewidth=1.2,
            label=f"Chosen ({DECISION_THRESHOLD})")
ax5.set_xlabel("Decision Threshold"); ax5.set_ylabel("Score")
ax5.set_title("Precision / Recall / F1 vs Decision Threshold", fontweight="bold")
ax5.legend()
st.pyplot(fig5)
plt.close(fig5)
