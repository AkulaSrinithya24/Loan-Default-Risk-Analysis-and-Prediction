"""
kpi.py
------
Key Performance Indicator (KPI) calculations for the loan portfolio.

Public API
----------
compute_portfolio_kpis(df)                    -> dict
compute_segment_kpis(df, segment_col)         -> pd.DataFrame
compute_kpi_trend(df, numeric_col, bins, labels) -> pd.DataFrame
print_portfolio_kpis(df)                      -> None
"""

import pandas as pd
import numpy as np

from src.config import (
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
)


def compute_portfolio_kpis(df: pd.DataFrame) -> dict:
    """Compute top-level portfolio KPIs.

    Returns
    -------
    dict with keys:
        total_loans, total_defaulted, total_non_defaulted,
        default_rate_pct, non_default_rate_pct,
        total_loan_value, avg_loan_amount, median_loan_amount,
        defaulted_loan_value, non_defaulted_loan_value,
        default_value_rate_pct,
        avg_credit_score, avg_income, avg_interest_rate,
        avg_dti_ratio, avg_loan_term_months,
        avg_months_employed
    """
    total          = len(df)
    defaulted      = int(df[TARGET_COLUMN].sum())
    non_defaulted  = total - defaulted

    total_value        = float(df["LoanAmount"].sum())
    defaulted_value    = float(df.loc[df[TARGET_COLUMN] == 1, "LoanAmount"].sum())
    non_default_value  = total_value - defaulted_value

    return {
        # Volume
        "total_loans":            total,
        "total_defaulted":        defaulted,
        "total_non_defaulted":    non_defaulted,
        # Rates
        "default_rate_pct":       round(defaulted / total * 100, 2),
        "non_default_rate_pct":   round(non_defaulted / total * 100, 2),
        # Value
        "total_loan_value":       round(total_value, 2),
        "avg_loan_amount":        round(df["LoanAmount"].mean(), 2),
        "median_loan_amount":     round(df["LoanAmount"].median(), 2),
        "defaulted_loan_value":   round(defaulted_value, 2),
        "non_defaulted_loan_value": round(non_default_value, 2),
        "default_value_rate_pct": round(defaulted_value / total_value * 100, 2),
        # Borrower profile
        "avg_credit_score":       round(df["CreditScore"].mean(), 1),
        "avg_income":             round(df["Income"].mean(), 2),
        "avg_interest_rate":      round(df["InterestRate"].mean(), 2),
        "avg_dti_ratio":          round(df["DTIRatio"].mean(), 3),
        "avg_loan_term_months":   round(df["LoanTerm"].mean(), 1),
        "avg_months_employed":    round(df["MonthsEmployed"].mean(), 1),
    }


def compute_segment_kpis(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """Compute KPIs broken down by a categorical segment column.

    Returns a DataFrame with one row per segment value, columns:
        segment_col, total_loans, defaults, default_rate_pct,
        avg_loan_amount, avg_income, avg_credit_score,
        avg_interest_rate, total_loan_value, default_loan_value,
        default_value_pct
    """
    rows = []
    for val, grp in df.groupby(segment_col):
        total   = len(grp)
        defs    = int(grp[TARGET_COLUMN].sum())
        total_v = float(grp["LoanAmount"].sum())
        def_v   = float(grp.loc[grp[TARGET_COLUMN] == 1, "LoanAmount"].sum())
        rows.append({
            segment_col:          val,
            "total_loans":        total,
            "defaults":           defs,
            "default_rate_pct":   round(defs / total * 100, 2),
            "avg_loan_amount":    round(grp["LoanAmount"].mean(), 0),
            "avg_income":         round(grp["Income"].mean(), 0),
            "avg_credit_score":   round(grp["CreditScore"].mean(), 1),
            "avg_interest_rate":  round(grp["InterestRate"].mean(), 2),
            "total_loan_value":   round(total_v, 0),
            "default_loan_value": round(def_v, 0),
            "default_value_pct":  round(def_v / total_v * 100, 2) if total_v > 0 else 0,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("default_rate_pct", ascending=False)
        .reset_index(drop=True)
    )


def compute_kpi_trend(
    df: pd.DataFrame,
    numeric_col: str,
    bins: list,
    labels: list,
) -> pd.DataFrame:
    """Compute default-rate KPIs across numeric bins (e.g. age bands, income quartiles).

    Parameters
    ----------
    df          : clean DataFrame
    numeric_col : column to bin (e.g. 'Age', 'Income')
    bins        : bin edges (list of numbers)
    labels      : bin labels (len = len(bins) - 1)

    Returns
    -------
    DataFrame with columns: [band_label, total, defaults, default_rate_pct,
                              avg_loan_amount, avg_credit_score]
    """
    tmp = df.copy()
    tmp["_band"] = pd.cut(tmp[numeric_col], bins=bins, labels=labels)
    rows = []
    for band, grp in tmp.groupby("_band", observed=True):
        rows.append({
            "band":              str(band),
            "total":             len(grp),
            "defaults":          int(grp[TARGET_COLUMN].sum()),
            "default_rate_pct":  round(grp[TARGET_COLUMN].mean() * 100, 2),
            "avg_loan_amount":   round(grp["LoanAmount"].mean(), 0),
            "avg_credit_score":  round(grp["CreditScore"].mean(), 1),
        })
    return pd.DataFrame(rows)


def print_portfolio_kpis(df: pd.DataFrame) -> None:
    """Pretty-print the top-level portfolio KPIs."""
    k = compute_portfolio_kpis(df)

    print("=" * 60)
    print("  PORTFOLIO KPIs")
    print("=" * 60)
    print(f"  Total Loans          : {k['total_loans']:>12,}")
    print(f"  Total Defaulted      : {k['total_defaulted']:>12,}  ({k['default_rate_pct']}%)")
    print(f"  Total Non-Defaulted  : {k['total_non_defaulted']:>12,}  ({k['non_default_rate_pct']}%)")
    print()
    print(f"  Total Loan Value     : ${k['total_loan_value']:>18,.2f}")
    print(f"  Avg  Loan Amount     : ${k['avg_loan_amount']:>18,.2f}")
    print(f"  Median Loan Amount   : ${k['median_loan_amount']:>18,.2f}")
    print(f"  Defaulted Loan Value : ${k['defaulted_loan_value']:>18,.2f}  ({k['default_value_rate_pct']}%)")
    print()
    print(f"  Avg Credit Score     : {k['avg_credit_score']:>12.1f}")
    print(f"  Avg Annual Income    : ${k['avg_income']:>18,.2f}")
    print(f"  Avg Interest Rate    : {k['avg_interest_rate']:>12.2f}%")
    print(f"  Avg DTI Ratio        : {k['avg_dti_ratio']:>12.3f}")
    print(f"  Avg Loan Term        : {k['avg_loan_term_months']:>12.1f} months")
    print(f"  Avg Months Employed  : {k['avg_months_employed']:>12.1f}")
    print("=" * 60)
