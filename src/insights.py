"""
insights.py
-----------
Generate plain-text business insights from the loan default dataset.
"""

import pandas as pd
import numpy as np

from src.config import TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def generate_insights(df: pd.DataFrame) -> list[dict]:
    """
    Return a list of insight dicts, each with:
      category, finding, evidence, recommendation, priority
    """
    insights = []

    # ── 1. Overall default rate ────────────────────────────────────────────────
    default_rate = df[TARGET_COLUMN].mean() * 100
    insights.append({
        "category":       "Portfolio Risk",
        "finding":        f"Overall default rate is {default_rate:.2f}%.",
        "evidence":       f"{int(df[TARGET_COLUMN].sum()):,} defaults out of {len(df):,} loans.",
        "recommendation": "Implement tighter credit screening for high-risk segments to reduce default rate below 10%.",
        "priority":       "High",
    })

    # ── 2. Credit score gap ────────────────────────────────────────────────────
    cs_def    = df.loc[df[TARGET_COLUMN] == 1, "CreditScore"].mean()
    cs_nodef  = df.loc[df[TARGET_COLUMN] == 0, "CreditScore"].mean()
    insights.append({
        "category":       "Credit Quality",
        "finding":        f"Defaulters have a mean credit score of {cs_def:.0f} vs {cs_nodef:.0f} for non-defaulters.",
        "evidence":       f"Gap of {abs(cs_nodef - cs_def):.0f} points in average credit score.",
        "recommendation": "Set a minimum credit score threshold (e.g., 600) as a hard eligibility criterion.",
        "priority":       "High",
    })

    # ── 3. Interest rate ───────────────────────────────────────────────────────
    ir_def   = df.loc[df[TARGET_COLUMN] == 1, "InterestRate"].mean()
    ir_nodef = df.loc[df[TARGET_COLUMN] == 0, "InterestRate"].mean()
    insights.append({
        "category":       "Loan Pricing",
        "finding":        f"Defaulted loans carry a higher average interest rate ({ir_def:.2f}%) vs non-defaulted ({ir_nodef:.2f}%).",
        "evidence":       f"Interest rate difference: {abs(ir_def - ir_nodef):.2f} percentage points.",
        "recommendation": "Review whether high interest rates are causing stress — consider capping rates or offering restructuring.",
        "priority":       "Medium",
    })

    # ── 4. DTI ratio ──────────────────────────────────────────────────────────
    dti_def   = df.loc[df[TARGET_COLUMN] == 1, "DTIRatio"].mean()
    dti_nodef = df.loc[df[TARGET_COLUMN] == 0, "DTIRatio"].mean()
    insights.append({
        "category":       "Debt Burden",
        "finding":        f"Defaulters show a higher mean DTI ratio ({dti_def:.3f}) than non-defaulters ({dti_nodef:.3f}).",
        "evidence":       f"DTI gap: {abs(dti_def - dti_nodef):.3f}.",
        "recommendation": "Enforce a DTI ratio ceiling (e.g., 0.45) in the underwriting policy.",
        "priority":       "High",
    })

    # ── 5. Employment type ────────────────────────────────────────────────────
    if "EmploymentType" in df.columns:
        emp_rates = (
            df.groupby("EmploymentType")[TARGET_COLUMN]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        worst_emp  = emp_rates.index[0]
        worst_rate = emp_rates.iloc[0]
        insights.append({
            "category":       "Employment Risk",
            "finding":        f"'{worst_emp}' borrowers have the highest default rate ({worst_rate:.2f}%).",
            "evidence":       emp_rates.to_string(),
            "recommendation": f"Apply stricter income-verification and collateral requirements for '{worst_emp}' applicants.",
            "priority":       "High",
        })

    # ── 6. Loan purpose ───────────────────────────────────────────────────────
    if "LoanPurpose" in df.columns:
        purpose_rates = (
            df.groupby("LoanPurpose")[TARGET_COLUMN]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        worst_purpose = purpose_rates.index[0]
        worst_p_rate  = purpose_rates.iloc[0]
        insights.append({
            "category":       "Loan Purpose",
            "finding":        f"'{worst_purpose}' loans have the highest default rate ({worst_p_rate:.2f}%).",
            "evidence":       purpose_rates.to_string(),
            "recommendation": f"Flag '{worst_purpose}' loan applications for enhanced due diligence.",
            "priority":       "Medium",
        })

    # ── 7. Income quartile ────────────────────────────────────────────────────
    df2 = df.copy()
    df2["IncomeQ"] = pd.qcut(df2["Income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    iq_rates = df2.groupby("IncomeQ")[TARGET_COLUMN].mean().mul(100)
    insights.append({
        "category":       "Income Segmentation",
        "finding":        f"Lowest income quartile (Q1) default rate: {iq_rates['Q1']:.2f}% vs Q4: {iq_rates['Q4']:.2f}%.",
        "evidence":       iq_rates.to_string(),
        "recommendation": "Offer smaller loan amounts and shorter terms to Q1 income borrowers to limit exposure.",
        "priority":       "Medium",
    })

    # ── 8. Predictive modelling ───────────────────────────────────────────────
    insights.append({
        "category":       "Predictive Modelling",
        "finding":        "Machine learning can predict default risk before loan disbursement.",
        "evidence":       "Random Forest model trained on all 16 features enables probability scoring per applicant.",
        "recommendation": "Deploy the ML model as a pre-approval scoring tool; flag applications above 40% predicted default probability for manual review.",
        "priority":       "High",
    })

    return insights
