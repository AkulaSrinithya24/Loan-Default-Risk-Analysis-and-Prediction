"""
dashboard/app.py
----------------
Main entry point for the Loan Default Risk Analytics Streamlit dashboard.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Loan Default Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar header ─────────────────────────────────────────────────────────────
st.sidebar.title("🏦 Loan Default Risk Analytics")
st.sidebar.markdown(
    """
    **IBM SkillsBuild**  
    Data Analytics Academic Internship  
    
    ---
    Navigate using the pages below.
    """
)

# ── Home page ──────────────────────────────────────────────────────────────────
st.title("Loan Default Risk Analysis and Prediction")
st.markdown("### IBM SkillsBuild Data Analytics Academic Internship")

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("📊 **Overview & KPIs**\n\nPortfolio-level key performance indicators and summary statistics.")
with col2:
    st.info("🔍 **EDA**\n\nExploratory data analysis — distributions, correlations, and patterns.")
with col3:
    st.info("⚠️ **Risk Analysis**\n\nRisk scoring, tier segmentation, and key drivers of default.")

col4, col5, col6 = st.columns(3)
with col4:
    st.info("🤖 **Model Performance**\n\nROC curve, confusion matrix, feature importances, and threshold analysis.")
with col5:
    st.info("🎯 **Prediction**\n\nPredict default probability for a new loan application.")
with col6:
    st.success(
        "**Dataset**\n\n"
        "255,347 loan records · 17 features · Binary target (Default 0/1)"
    )

st.markdown("---")
st.markdown(
    """
    **Use the sidebar to navigate between pages.**  
    All analysis is powered by `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`.  
    The predictive model is a Random Forest classifier trained on 80% of the data.
    """
)
