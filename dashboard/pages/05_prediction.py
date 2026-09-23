"""
dashboard/pages/05_prediction.py
----------------------------------
Single-loan default prediction interface.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.config import MODEL_PATH, DECISION_THRESHOLD

st.set_page_config(page_title="Prediction", page_icon="🎯", layout="wide")
st.title("🎯 Loan Default Prediction")
st.markdown("Enter the details of a loan application to predict the default probability.")
st.markdown("---")

# ── Check model exists ─────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.warning(
        "No trained model found. Please run the training pipeline first:\n\n"
        "```\npython run_pipeline.py\n```"
    )
    st.stop()


@st.cache_resource
def load_model():
    import joblib
    return joblib.load(MODEL_PATH)

pipeline = load_model()

# ── Input form ─────────────────────────────────────────────────────────────────
st.subheader("Loan Application Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Borrower Profile**")
    age             = st.slider("Age",              18,  69,  35)
    income          = st.number_input("Annual Income ($)", 15000, 150000, 65000, step=1000)
    credit_score    = st.slider("Credit Score",     300, 849, 600)
    months_employed = st.slider("Months Employed",  0,   119, 48)
    education       = st.selectbox("Education",
                        ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Employment Type",
                        ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital_status  = st.selectbox("Marital Status",
                        ["Single", "Married", "Divorced"])

with col2:
    st.markdown("**Loan Details**")
    loan_amount     = st.number_input("Loan Amount ($)",  5000, 250000, 100000, step=1000)
    interest_rate   = st.slider("Interest Rate (%)",  2.0, 25.0, 10.0, step=0.1)
    loan_term       = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], index=2)
    dti_ratio       = st.slider("DTI Ratio",          0.10, 0.90, 0.40, step=0.01)
    num_credit_lines = st.slider("Num Credit Lines",   1, 4, 2)
    loan_purpose    = st.selectbox("Loan Purpose",
                        ["Auto", "Business", "Education", "Home", "Other"])

with col3:
    st.markdown("**Additional Flags**")
    has_mortgage    = st.radio("Has Mortgage?",   ["Yes", "No"], index=1)
    has_dependents  = st.radio("Has Dependents?", ["Yes", "No"], index=1)
    has_cosigner    = st.radio("Has Co-Signer?",  ["Yes", "No"], index=1)

st.markdown("---")

# ── Predict button ─────────────────────────────────────────────────────────────
if st.button("🔍 Predict Default Risk", type="primary", use_container_width=True):

    input_data = pd.DataFrame([{
        "Age":             age,
        "Income":          income,
        "LoanAmount":      loan_amount,
        "CreditScore":     credit_score,
        "MonthsEmployed":  months_employed,
        "NumCreditLines":  num_credit_lines,
        "InterestRate":    interest_rate,
        "LoanTerm":        loan_term,
        "DTIRatio":        dti_ratio,
        "Education":       education,
        "EmploymentType":  employment_type,
        "MaritalStatus":   marital_status,
        "HasMortgage":     1 if has_mortgage == "Yes" else 0,
        "HasDependents":   1 if has_dependents == "Yes" else 0,
        "LoanPurpose":     loan_purpose,
        "HasCoSigner":     1 if has_cosigner == "Yes" else 0,
    }])

    prob   = float(pipeline.predict_proba(input_data)[0, 1])
    pred   = int(prob >= DECISION_THRESHOLD)

    if prob >= 0.60:
        risk_label = "Very High"
        color = "#8e1414"
    elif prob >= 0.40:
        risk_label = "High"
        color = "#e05c5c"
    elif prob >= 0.20:
        risk_label = "Medium"
        color = "#f39c12"
    else:
        risk_label = "Low"
        color = "#27ae60"

    st.markdown("---")
    st.subheader("Prediction Result")

    r1, r2, r3 = st.columns(3)
    r1.metric("Default Probability", f"{prob*100:.1f}%")
    r2.metric("Prediction",          "DEFAULT" if pred == 1 else "NO DEFAULT")
    r3.metric("Risk Tier",           risk_label)

    # Visual gauge
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 2.2))
    # Background gradient bar
    gradient = np.linspace(0, 1, 500).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap="RdYlGn_r",
              extent=[0, 1, 0, 1], alpha=0.85)
    # Pointer
    ax.axvline(prob, color="black", linewidth=3)
    ax.text(prob, 1.12, f"{prob*100:.1f}%", ha="center", va="bottom",
            fontsize=13, fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_title("Default Probability Gauge", fontweight="bold", pad=14)
    ax.axvline(DECISION_THRESHOLD, color="white", linewidth=1.5,
               linestyle="--", alpha=0.8)
    ax.text(DECISION_THRESHOLD + 0.01, 0.5, f"Threshold\n{DECISION_THRESHOLD*100:.0f}%",
            color="white", fontsize=8, va="center")
    st.pyplot(fig)
    plt.close(fig)

    # Explanation
    st.markdown("---")
    st.subheader("Interpretation")
    if pred == 1:
        st.error(
            f"⚠️ This application is predicted to **DEFAULT** with a probability of "
            f"**{prob*100:.1f}%** (threshold: {DECISION_THRESHOLD*100:.0f}%).  \n"
            f"Risk tier: **{risk_label}**.  \n"
            "Consider requiring a co-signer, reducing the loan amount, or declining."
        )
    else:
        st.success(
            f"✅ This application is predicted to **NOT DEFAULT** with a default probability of "
            f"**{prob*100:.1f}%** (threshold: {DECISION_THRESHOLD*100:.0f}%).  \n"
            f"Risk tier: **{risk_label}**.  \n"
            "Standard approval process recommended."
        )

    # Input summary table
    with st.expander("View input data submitted"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}),
                     use_container_width=True)
