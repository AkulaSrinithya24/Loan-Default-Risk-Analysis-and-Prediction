"""
dashboard/pages/02_eda.py
--------------------------
Exploratory Data Analysis page.
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

from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.eda import (
    plot_numeric_by_default,
    plot_all_numeric_by_default,
    plot_correlation_heatmap,
    plot_default_rate_by_category,
    plot_default_rate_heatmap,
    plot_categorical_default_rates,
    plot_binary_default_rates,
    get_default_rates_by_category,
)
from src.config import (
    RAW_DATA_PATH, TARGET_COLUMN,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES,
)

st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")
st.title("🔍 Exploratory Data Analysis")
st.markdown("---")


@st.cache_data
def get_data():
    return clean_data(load_data(RAW_DATA_PATH))

df = get_data()

# ── Section 1: Numeric distributions ──────────────────────────────────────────
st.subheader("1. Numeric Feature Distributions by Default Status")
st.markdown("Select a feature for a detailed KDE + box plot comparison.")

feat = st.selectbox("Select numeric feature:", NUMERIC_FEATURES, index=0)
fig = plot_numeric_by_default(df, feat)
st.pyplot(fig)
plt.close(fig)

with st.expander("Show all numeric features at once"):
    fig2 = plot_all_numeric_by_default(df)
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown("---")

# ── Section 2: Correlation heatmap ────────────────────────────────────────────
st.subheader("2. Correlation Heatmap")
fig3 = plot_correlation_heatmap(df)
st.pyplot(fig3)
plt.close(fig3)

st.markdown("**Correlation with Default (ranked):**")
corr_target = (
    df[NUMERIC_FEATURES + [TARGET_COLUMN]]
    .corr()[TARGET_COLUMN]
    .drop(TARGET_COLUMN)
    .abs()
    .sort_values(ascending=False)
    .round(4)
    .reset_index()
)
corr_target.columns = ["Feature", "|Pearson r| with Default"]
st.dataframe(corr_target, use_container_width=True)

st.markdown("---")

# ── Section 3: Categorical default rates ──────────────────────────────────────
st.subheader("3. Default Rate by Categorical Feature")
cat_choice = st.selectbox("Select categorical feature:", CATEGORICAL_FEATURES, index=0)
fig4 = plot_default_rate_by_category(df, cat_choice)
st.pyplot(fig4)
plt.close(fig4)

rates_table = get_default_rates_by_category(df, cat_choice)
st.dataframe(
    rates_table.style.format({
        "default_rate_pct": "{:.2f}%",
        "total": "{:,}",
        "defaults": "{:,}",
    }),
    use_container_width=True,
)

st.markdown("---")

# ── Section 4: Binary features ────────────────────────────────────────────────
st.subheader("4. Default Rate by Binary Features")
fig5 = plot_binary_default_rates(df)
st.pyplot(fig5)
plt.close(fig5)

st.markdown("---")

# ── Section 5: Cross-feature heatmap ──────────────────────────────────────────
st.subheader("5. Cross-Feature Default Rate Heatmap")
all_cats = CATEGORICAL_FEATURES
col1, col2 = st.columns(2)
with col1:
    cat1 = st.selectbox("Row feature:", all_cats, index=1)  # EmploymentType
with col2:
    remaining = [c for c in all_cats if c != cat1]
    cat2 = st.selectbox("Column feature:", remaining, index=0)

fig6 = plot_default_rate_heatmap(df, cat1, cat2)
st.pyplot(fig6)
plt.close(fig6)

st.markdown("---")

# ── Section 6: Age & Income binned rates ──────────────────────────────────────
st.subheader("6. Default Rate by Age Band & Income Quartile")
c_left, c_right = st.columns(2)

with c_left:
    tmp = df.copy()
    tmp["AgeBand"] = pd.cut(tmp["Age"], bins=[17, 25, 35, 45, 55, 70],
                            labels=["18-25", "26-35", "36-45", "46-55", "56-69"])
    age_rates = get_default_rates_by_category(tmp, "AgeBand")
    fig7, ax7 = plt.subplots(figsize=(6, 3.5))
    bars = ax7.bar(age_rates["AgeBand"].astype(str),
                   age_rates["default_rate_pct"],
                   color="#e05c5c", edgecolor="white")
    ax7.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax7.set_title("Default Rate by Age Band", fontweight="bold")
    ax7.set_ylabel("Default Rate (%)")
    ax7.set_ylim(0, age_rates["default_rate_pct"].max() * 1.3)
    st.pyplot(fig7)
    plt.close(fig7)

with c_right:
    tmp2 = df.copy()
    tmp2["IncomeQ"] = pd.qcut(tmp2["Income"], q=4,
                               labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
    inc_rates = get_default_rates_by_category(tmp2, "IncomeQ")
    inc_rates = inc_rates.sort_values("IncomeQ")
    fig8, ax8 = plt.subplots(figsize=(6, 3.5))
    bars = ax8.bar(inc_rates["IncomeQ"].astype(str),
                   inc_rates["default_rate_pct"],
                   color="#7c5cd8", edgecolor="white")
    ax8.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax8.set_title("Default Rate by Income Quartile", fontweight="bold")
    ax8.set_ylabel("Default Rate (%)")
    ax8.set_ylim(0, inc_rates["default_rate_pct"].max() * 1.3)
    st.pyplot(fig8)
    plt.close(fig8)
