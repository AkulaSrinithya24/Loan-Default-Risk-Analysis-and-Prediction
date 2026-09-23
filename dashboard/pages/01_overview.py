"""
dashboard/pages/01_overview.py
-------------------------------
Portfolio Overview & KPI Dashboard page.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.kpi import compute_portfolio_kpis, compute_segment_kpis, compute_kpi_trend
from src.config import RAW_DATA_PATH, TARGET_COLUMN, CATEGORICAL_FEATURES

st.set_page_config(page_title="Overview & KPIs", page_icon="📊", layout="wide")
st.title("📊 Portfolio Overview & KPIs")
st.markdown("---")


# ── Cached data loading ────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return clean_data(load_data(RAW_DATA_PATH))


df = get_data()
kpis = compute_portfolio_kpis(df)

# ── KPI Metric cards ──────────────────────────────────────────────────────────
st.subheader("Portfolio KPIs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Loans",        f"{kpis['total_loans']:,}")
c2.metric("Default Rate",       f"{kpis['default_rate_pct']}%",
          delta=f"{kpis['total_defaulted']:,} loans defaulted", delta_color="inverse")
c3.metric("Total Loan Value",   f"${kpis['total_loan_value']/1e9:.2f}B")
c4.metric("Defaulted Value",    f"${kpis['defaulted_loan_value']/1e9:.2f}B",
          delta=f"{kpis['default_value_rate_pct']}% of portfolio", delta_color="inverse")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Avg Loan Amount",    f"${kpis['avg_loan_amount']:,.0f}")
c6.metric("Avg Credit Score",   f"{kpis['avg_credit_score']:.0f}")
c7.metric("Avg Interest Rate",  f"{kpis['avg_interest_rate']}%")
c8.metric("Avg DTI Ratio",      f"{kpis['avg_dti_ratio']:.3f}")

st.markdown("---")

# ── Class balance ──────────────────────────────────────────────────────────────
st.subheader("Target Variable Distribution")
col_a, col_b = st.columns([1, 1])

with col_a:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["No Default", "Default"], counts.values,
           color=["#4a90d9", "#e05c5c"], edgecolor="white", linewidth=0.8)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 300, f"{v:,}", ha="center", fontsize=10)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_title("Loan Default Counts", fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)

with col_b:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.pie(counts.values, labels=["No Default", "Default"],
           autopct="%1.1f%%", colors=["#4a90d9", "#e05c5c"],
           startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax.set_title("Default Class Balance", fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# ── Segment KPIs ───────────────────────────────────────────────────────────────
st.subheader("Segment KPIs")
segment_choice = st.selectbox(
    "Select segment:",
    CATEGORICAL_FEATURES + ["HasMortgage", "HasDependents", "HasCoSigner"],
    index=1,
)

seg_kpis = compute_segment_kpis(df, segment_choice)

# Bar chart
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(
    seg_kpis[segment_choice].astype(str),
    seg_kpis["default_rate_pct"],
    color="#e05c5c", edgecolor="white", linewidth=0.8,
)
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_title(f"Default Rate by {segment_choice}", fontweight="bold")
ax.set_ylabel("Default Rate (%)")
ax.set_ylim(0, seg_kpis["default_rate_pct"].max() * 1.3)
ax.tick_params(axis="x", rotation=20)
st.pyplot(fig)
plt.close(fig)

# Data table
st.dataframe(seg_kpis.style.format({
    "default_rate_pct": "{:.2f}%",
    "avg_loan_amount":  "${:,.0f}",
    "avg_income":       "${:,.0f}",
    "total_loan_value": "${:,.0f}",
    "default_loan_value": "${:,.0f}",
    "default_value_pct": "{:.2f}%",
}), use_container_width=True)

st.markdown("---")

# ── Age band trend ─────────────────────────────────────────────────────────────
st.subheader("Default Rate by Age Band")
age_trend = compute_kpi_trend(
    df, "Age",
    bins=[17, 25, 35, 45, 55, 70],
    labels=["18-25", "26-35", "36-45", "46-55", "56-69"],
)
fig, ax = plt.subplots(figsize=(8, 3.5))
bars = ax.bar(age_trend["band"], age_trend["default_rate_pct"],
              color="#7c5cd8", edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_ylabel("Default Rate (%)")
ax.set_title("Default Rate Trend by Age Band", fontweight="bold")
ax.set_ylim(0, age_trend["default_rate_pct"].max() * 1.3)
st.pyplot(fig)
plt.close(fig)
