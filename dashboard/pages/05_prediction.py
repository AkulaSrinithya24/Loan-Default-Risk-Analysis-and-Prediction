"""
dashboard/pages/05_prediction.py
---------------------------------
Loan Default Prediction Interface

This page provides:
- Loan application input form
- Machine-learning default probability
- Model verdict
- Risk-tier classification
- Simple probability visualization
- Input/data-quality checks
- Risk indicators
- Supporting application context
- Risk-tier explanation
- Submitted application details

Important:
The machine-learning prediction logic is unchanged.
The additional messages shown on this page are only
business-rule/data-quality observations and do not
modify the model prediction.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from src.config import (
    MODEL_PATH,
    DEPLOYMENT_MODEL_PATH,
    DECISION_THRESHOLD,
)

from dashboard.theme import page_header, section_title, show_table


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Prediction",
    page_icon="🎯",
    layout="wide",
)


# =============================================================================
# PAGE HEADER
# =============================================================================

page_header(
    "Loan Default Prediction",
    "Enter loan application details to estimate default probability and risk classification.",
)


# =============================================================================
# MODEL LOADING
# =============================================================================

if MODEL_PATH.exists():
    _model_path = MODEL_PATH

elif DEPLOYMENT_MODEL_PATH.exists():
    _model_path = DEPLOYMENT_MODEL_PATH

    st.info(
        "The deployment model is currently being used. "
        "The prediction functionality remains fully available."
    )

else:
    st.error(
        "No trained model was found. Please make sure the model artifact "
        "has been generated before using the prediction page."
    )
    st.stop()


@st.cache_resource
def load_model():
    import joblib

    return joblib.load(_model_path)


pipeline = load_model()


# =============================================================================
# INPUT SECTION
# =============================================================================

section_title("Loan Application Details")

st.caption(
    "Enter the applicant and loan details below. "
    "The trained machine-learning model will use these inputs to estimate "
    "the probability of loan default."
)

st.markdown("")


# =============================================================================
# THREE-COLUMN INPUT FORM
# =============================================================================

col1, col2, col3 = st.columns(3)


# =============================================================================
# BORROWER PROFILE
# =============================================================================

with col1:

    st.markdown("#### Borrower Profile")

    age = st.slider(
        "Age",
        min_value=18,
        max_value=69,
        value=35,
    )

    income = st.number_input(
        "Annual Income ($)",
        min_value=15000,
        max_value=150000,
        value=65000,
        step=1000,
    )

    credit_score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=849,
        value=600,
    )

    months_employed = st.slider(
        "Months Employed",
        min_value=0,
        max_value=119,
        value=48,
    )

    education = st.selectbox(
        "Education",
        [
            "High School",
            "Bachelor's",
            "Master's",
            "PhD",
        ],
    )

    employment_type = st.selectbox(
        "Employment Type",
        [
            "Full-time",
            "Part-time",
            "Self-employed",
            "Unemployed",
        ],
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced",
        ],
    )


# =============================================================================
# LOAN DETAILS
# =============================================================================

with col2:

    st.markdown("#### Loan Details")

    loan_amount = st.number_input(
        "Loan Amount ($)",
        min_value=5000,
        max_value=250000,
        value=100000,
        step=1000,
    )

    interest_rate = st.slider(
        "Interest Rate (%)",
        min_value=2.0,
        max_value=25.0,
        value=10.0,
        step=0.1,
    )

    loan_term = st.selectbox(
        "Loan Term (months)",
        [12, 24, 36, 48, 60],
        index=2,
    )

    dti_ratio = st.slider(
        "DTI Ratio",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.01,
    )

    num_credit_lines = st.slider(
        "Number of Credit Lines",
        min_value=1,
        max_value=4,
        value=2,
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        [
            "Auto",
            "Business",
            "Education",
            "Home",
            "Other",
        ],
    )


# =============================================================================
# ADDITIONAL FLAGS
# =============================================================================

with col3:

    st.markdown("#### Additional Information")

    st.markdown("**Existing Mortgage**")

    has_mortgage = st.radio(
        "Has Mortgage?",
        ["Yes", "No"],
        index=1,
        label_visibility="collapsed",
    )

    st.markdown("**Dependents**")

    has_dependents = st.radio(
        "Has Dependents?",
        ["Yes", "No"],
        index=1,
        label_visibility="collapsed",
    )

    st.markdown("**Co-Signer**")

    has_cosigner = st.radio(
        "Has Co-Signer?",
        ["Yes", "No"],
        index=1,
        label_visibility="collapsed",
    )


# =============================================================================
# INPUT NOTE
# =============================================================================

st.info(
    "Tip: Make sure the information entered is accurate. "
    "The prediction is based on the values provided in this form."
)


# =============================================================================
# PREDICTION BUTTON
# =============================================================================

st.markdown("")

run_prediction = st.button(
    "Run Default Risk Assessment",
    type="primary",
    use_container_width=True,
)


# =============================================================================
# PREDICTION
# =============================================================================

if run_prediction:

    # =========================================================================
    # PREPARE INPUT DATA
    # =========================================================================

    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Income": income,
                "LoanAmount": loan_amount,
                "CreditScore": credit_score,
                "MonthsEmployed": months_employed,
                "NumCreditLines": num_credit_lines,
                "InterestRate": interest_rate,
                "LoanTerm": loan_term,
                "DTIRatio": dti_ratio,
                "Education": education,
                "EmploymentType": employment_type,
                "MaritalStatus": marital_status,
                "HasMortgage": 1 if has_mortgage == "Yes" else 0,
                "HasDependents": 1 if has_dependents == "Yes" else 0,
                "LoanPurpose": loan_purpose,
                "HasCoSigner": 1 if has_cosigner == "Yes" else 0,
            }
        ]
    )


    # =========================================================================
    # MACHINE LEARNING PREDICTION
    # =========================================================================

    prob = float(
        pipeline.predict_proba(input_data)[0, 1]
    )

    pred = int(
        prob >= DECISION_THRESHOLD
    )


    # =========================================================================
    # RISK TIER
    # =========================================================================

    if prob >= 0.60:
        risk_label = "Very High"

    elif prob >= 0.40:
        risk_label = "High"

    elif prob >= 0.20:
        risk_label = "Medium"

    else:
        risk_label = "Low"


    # =========================================================================
    # ASSESSMENT RESULT
    # =========================================================================

    st.markdown("---")

    section_title("Assessment Result")

    st.caption(
        "The result below is generated by the trained machine-learning model "
        "using the information entered above."
    )

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Default Probability",
            f"{prob * 100:.1f}%",
        )

    with result_col2:
        st.metric(
            "Model Verdict",
            "DEFAULT" if pred == 1 else "NO DEFAULT",
        )

    with result_col3:
        st.metric(
            "Risk Tier",
            risk_label,
        )


    # =========================================================================
    # SHORT MODEL RESULT MESSAGE
    # =========================================================================

    st.markdown("")

    if pred == 1:

        st.error(
            f"The model estimates a default probability of "
            f"**{prob * 100:.1f}%**, which is at or above the "
            f"configured decision threshold of "
            f"**{DECISION_THRESHOLD * 100:.0f}%**."
        )

    else:

        st.success(
            f"The model estimates a default probability of "
            f"**{prob * 100:.1f}%**, which is below the "
            f"configured decision threshold of "
            f"**{DECISION_THRESHOLD * 100:.0f}%**."
        )


    # =========================================================================
    # PROBABILITY OVERVIEW
    # =========================================================================

    section_title("Default Probability")

    st.progress(
        min(max(prob, 0.0), 1.0),
        text=f"Estimated default probability: {prob * 100:.1f}%",
    )

    probability_col1, probability_col2, probability_col3 = st.columns(3)

    with probability_col1:
        st.caption("0% — Lower estimated risk")

    with probability_col2:
        st.caption(
            f"Decision threshold: {DECISION_THRESHOLD * 100:.0f}%"
        )

    with probability_col3:
        st.caption("100% — Higher estimated risk")


    # =========================================================================
    # APPLICATION REVIEW
    # =========================================================================

    section_title("Application Review")

    st.caption(
        "These observations provide additional context about the entered "
        "application. They do not change the machine-learning prediction."
    )


    # =========================================================================
    # REVIEW LISTS
    # =========================================================================

    data_quality_checks = []

    risk_indicators = []

    supporting_indicators = []


    # =========================================================================
    # CALCULATED VALUES
    # =========================================================================

    loan_income_ratio = (
        loan_amount / income
        if income > 0
        else 0
    )

    monthly_income = income / 12

    approximate_monthly_loan = (
        loan_amount / loan_term
        if loan_term > 0
        else 0
    )

    loan_to_monthly_income = (
        approximate_monthly_loan / monthly_income
        if monthly_income > 0
        else 0
    )


    # Keep calculated value available without changing model output.
    _ = loan_to_monthly_income


    # =========================================================================
    # INPUT / DATA QUALITY CHECKS
    # =========================================================================

    if employment_type == "Unemployed" and months_employed > 0:

        data_quality_checks.append(
            f"Employment is marked **Unemployed**, but "
            f"**{months_employed} months** of employment history are entered. "
            f"Please verify these values."
        )


    if employment_type == "Unemployed" and income > 0:

        data_quality_checks.append(
            f"Annual income is **${income:,.0f}** while employment status is "
            f"**Unemployed**. Confirm the source of the reported income."
        )


    if loan_income_ratio >= 2.0:

        data_quality_checks.append(
            f"The requested loan is about **{loan_income_ratio:.1f}× "
            f"annual income**. This is a relatively large loan compared "
            f"with the reported income."
        )

    elif loan_income_ratio >= 1.25:

        data_quality_checks.append(
            f"The requested loan is about **{loan_income_ratio:.1f}× "
            f"annual income**. Review the loan amount together with income."
        )


    # =========================================================================
    # RISK INDICATORS
    # =========================================================================

    if dti_ratio >= 0.50:

        risk_indicators.append(
            f"DTI ratio is **{dti_ratio:.2f}**, indicating relatively high "
            f"debt obligations compared with income."
        )

    elif dti_ratio >= 0.40:

        risk_indicators.append(
            f"DTI ratio is **{dti_ratio:.2f}**, which is elevated and should "
            f"be considered together with income and loan amount."
        )


    if credit_score < 580:

        risk_indicators.append(
            f"Credit score is **{credit_score}**, which is low within the "
            f"configured input range and may warrant additional review."
        )

    elif credit_score < 650:

        risk_indicators.append(
            f"Credit score is **{credit_score}**, which may warrant "
            f"additional review alongside other risk factors."
        )

    elif credit_score < 700:

        supporting_indicators.append(
            f"Credit score is **{credit_score}**. Consider it together "
            f"with the other application characteristics."
        )


    if loan_income_ratio >= 2.0:

        risk_indicators.append(
            f"Loan amount is about **{loan_income_ratio:.1f}× annual income**, "
            f"creating a relatively large loan-to-income relationship."
        )

    elif loan_income_ratio >= 1.25:

        risk_indicators.append(
            f"Loan amount is about **{loan_income_ratio:.1f}× annual income**. "
            f"This relationship should be reviewed with the other inputs."
        )


    if interest_rate >= 18:

        risk_indicators.append(
            f"Interest rate is **{interest_rate:.1f}%**, which is high "
            f"within the available input range."
        )

    elif interest_rate >= 12:

        supporting_indicators.append(
            f"Interest rate is **{interest_rate:.1f}%**, which is relatively "
            f"high within the available input range."
        )


    if months_employed <= 12:

        risk_indicators.append(
            f"Employment history is only **{months_employed} months**. "
            f"Employment stability may need closer review."
        )

    elif months_employed <= 24:

        supporting_indicators.append(
            f"Employment history of **{months_employed} months** represents "
            f"a relatively short employment history."
        )

    else:

        supporting_indicators.append(
            f"Employment history of **{months_employed} months** is reported."
        )


    if has_cosigner == "No" and (
        credit_score < 650
        or dti_ratio >= 0.40
        or loan_income_ratio >= 1.25
    ):

        risk_indicators.append(
            "No co-signer is reported while other review indicators are "
            "present. This may warrant closer review."
        )


    if has_mortgage == "Yes":

        supporting_indicators.append(
            "An existing mortgage is reported. Consider this obligation "
            "together with the DTI ratio."
        )


    if has_dependents == "Yes":

        supporting_indicators.append(
            "Dependents are reported. Household obligations may provide "
            "additional context when reviewing affordability."
        )


    if num_credit_lines <= 1:

        supporting_indicators.append(
            f"Number of credit lines: **{num_credit_lines}**. This provides "
            f"limited context about existing credit history."
        )

    elif num_credit_lines >= 4:

        supporting_indicators.append(
            f"Number of credit lines: **{num_credit_lines}**. Multiple "
            f"credit lines should be considered with the DTI ratio and "
            f"credit score."
        )


    # =========================================================================
    # MODEL RISK-TIER CHECK
    # =========================================================================

    if risk_label == "Medium":

        risk_indicators.insert(
            0,
            f"The model places this application in the **Medium** risk tier "
            f"with a predicted probability of **{prob * 100:.1f}%**. "
            f"The result is below the {DECISION_THRESHOLD * 100:.0f}% "
            f"decision threshold, but the application deserves closer review."
        )

    elif risk_label == "High":

        risk_indicators.insert(
            0,
            f"The model places this application in the **High** risk tier "
            f"with a predicted probability of **{prob * 100:.1f}%**, "
            f"which is at or above the {DECISION_THRESHOLD * 100:.0f}% "
            f"decision threshold."
        )

    elif risk_label == "Very High":

        risk_indicators.insert(
            0,
            f"The model places this application in the **Very High** risk tier "
            f"with a predicted probability of **{prob * 100:.1f}%**. "
            f"The probability is well above the "
            f"{DECISION_THRESHOLD * 100:.0f}% decision threshold."
        )


    # =========================================================================
    # DISPLAY DATA QUALITY REVIEW
    # =========================================================================

    review_col1, review_col2 = st.columns(2)


    # =========================================================================
    # INPUT & DATA QUALITY
    # =========================================================================

    with review_col1:

        st.markdown("#### Input & Data Quality")

        st.caption(
            "Checks for unusual or potentially inconsistent combinations "
            "of entered values."
        )

        if data_quality_checks:

            message = "\n\n".join(
                [
                    f"- {item}"
                    for item in data_quality_checks
                ]
            )

            st.warning(message)

        else:

            st.success(
                "No major input consistency issues were detected."
            )


    # =========================================================================
    # RISK INDICATORS
    # =========================================================================

    with review_col2:

        st.markdown("#### Risk Indicators")

        st.caption(
            "Important application characteristics that may need "
            "additional attention."
        )

        if risk_indicators:

            message = "\n\n".join(
                [
                    f"- {item}"
                    for item in risk_indicators
                ]
            )

            st.warning(message)

        else:

            st.success(
                "No major risk indicators were triggered by the "
                "configured review checks."
            )


    # =========================================================================
    # SUPPORTING APPLICATION CONTEXT
    # =========================================================================

    if supporting_indicators:

        st.markdown("")

        st.markdown("#### Supporting Application Context")

        st.caption(
            "Additional information that can help the user understand "
            "the application without changing the model result."
        )

        message = "\n\n".join(
            [
                f"- {item}"
                for item in supporting_indicators
            ]
        )

        st.info(message)


    # =========================================================================
    # MODEL INTERPRETATION
    # =========================================================================

    section_title("Model Interpretation")

    if pred == 1:

        st.error(
            f"**Default predicted** — The estimated default probability is "
            f"**{prob * 100:.1f}%**, which meets or exceeds the "
            f"{DECISION_THRESHOLD * 100:.0f}% model decision threshold."
        )

    else:

        st.success(
            f"**No default predicted** — The estimated default probability is "
            f"**{prob * 100:.1f}%**, which is below the "
            f"{DECISION_THRESHOLD * 100:.0f}% model decision threshold."
        )


    interpretation_col1, interpretation_col2, interpretation_col3 = st.columns(3)

    with interpretation_col1:

        st.metric(
            "Default Probability",
            f"{prob * 100:.1f}%",
        )

    with interpretation_col2:

        st.metric(
            "Decision Threshold",
            f"{DECISION_THRESHOLD * 100:.0f}%",
        )

    with interpretation_col3:

        st.metric(
            "Risk Tier",
            risk_label,
        )


    st.caption(
        "The risk tier is a classification based on the predicted probability. "
        "It does not replace the underlying machine-learning prediction."
    )


    # =========================================================================
    # RISK TIER DEFINITIONS
    # =========================================================================

    with st.expander("Understand the risk tiers"):

        risk_tier_data = pd.DataFrame(
            {
                "Risk Tier": [
                    "Low",
                    "Medium",
                    "High",
                    "Very High",
                ],
                "Default Probability": [
                    "Below 20%",
                    "20% – 39.9%",
                    "40% – 59.9%",
                    "60% and above",
                ],
            }
        )

        st.table(risk_tier_data)

        st.caption(
            f"Model decision threshold: "
            f"{DECISION_THRESHOLD * 100:.0f}%"
        )


    # =========================================================================
    # SUBMITTED APPLICATION DATA
    # =========================================================================

    with st.expander("View submitted application details"):

        display_input = (
            input_data.T
            .rename(columns={0: "Value"})
            .astype(str)
        )

        show_table(display_input)