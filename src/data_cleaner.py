"""
data_cleaner.py
---------------
Data quality checks and cleaning pipeline for the loan default dataset.

Public API
----------
check_data_quality(df)   -> dict   (quality report)
clean_data(df)           -> pd.DataFrame  (cleaned, analysis-ready)
print_quality_report(df) -> None
"""

import pandas as pd
import numpy as np

from src.config import (
    TARGET_COLUMN,
    ID_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
)

# ── Expected valid values for categorical columns ──────────────────────────────
VALID_VALUES = {
    "Education":      {"High School", "Bachelor's", "Master's", "PhD"},
    "EmploymentType": {"Full-time", "Part-time", "Self-employed", "Unemployed"},
    "MaritalStatus":  {"Single", "Married", "Divorced"},
    "LoanPurpose":    {"Auto", "Business", "Education", "Home", "Other"},
    "HasMortgage":    {"Yes", "No"},
    "HasDependents":  {"Yes", "No"},
    "HasCoSigner":    {"Yes", "No"},
}

# ── Sensible numeric bounds (domain knowledge) ────────────────────────────────
NUMERIC_BOUNDS = {
    "Age":            (18,    100),
    "Income":         (0,     1_000_000),
    "LoanAmount":     (100,   10_000_000),
    "CreditScore":    (300,   850),
    "MonthsEmployed": (0,     600),
    "NumCreditLines": (0,     100),
    "InterestRate":   (0,     100),
    "LoanTerm":       (1,     360),
    "DTIRatio":       (0.0,   5.0),
}


# ── Quality checks ─────────────────────────────────────────────────────────────

def check_data_quality(df: pd.DataFrame) -> dict:
    """Run a comprehensive quality check and return a structured report.

    Checks performed
    ----------------
    1. Missing values per column
    2. Duplicate rows (excluding LoanID)
    3. Invalid categorical values
    4. Out-of-range numeric values
    5. Target column validity (only 0 / 1)
    6. Negative values in numeric columns that must be non-negative
    """
    report = {}
    total = len(df)

    # 1. Missing values
    missing = df.isnull().sum()
    report["missing_values"] = missing[missing > 0].to_dict()

    # 2. Duplicates
    dup_mask = df.drop(columns=[ID_COLUMN], errors="ignore").duplicated()
    report["duplicate_rows"] = int(dup_mask.sum())

    # 3. Invalid categoricals
    invalid_cats = {}
    for col, valid in VALID_VALUES.items():
        if col in df.columns:
            bad = df[~df[col].astype(str).str.strip().isin(valid)][col]
            if len(bad) > 0:
                invalid_cats[col] = {"count": len(bad), "values": bad.unique().tolist()}
    report["invalid_categorical_values"] = invalid_cats

    # 4. Out-of-range numerics
    out_of_range = {}
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col in df.columns:
            bad = df[(df[col] < lo) | (df[col] > hi)]
            if len(bad) > 0:
                out_of_range[col] = {
                    "count": len(bad),
                    "min":   float(df[col].min()),
                    "max":   float(df[col].max()),
                    "bounds": (lo, hi),
                }
    report["out_of_range_numerics"] = out_of_range

    # 5. Target validity
    invalid_target = df[~df[TARGET_COLUMN].isin([0, 1])]
    report["invalid_target_values"] = int(len(invalid_target))

    # 6. Negative values in strictly non-negative columns
    non_neg_cols = ["Income", "LoanAmount", "CreditScore", "MonthsEmployed",
                    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]
    negatives = {}
    for col in non_neg_cols:
        if col in df.columns:
            n = int((df[col] < 0).sum())
            if n > 0:
                negatives[col] = n
    report["negative_values"] = negatives

    # Summary flag
    issues = (
        len(report["missing_values"])
        + report["duplicate_rows"]
        + len(report["invalid_categorical_values"])
        + len(report["out_of_range_numerics"])
        + report["invalid_target_values"]
        + len(report["negative_values"])
    )
    report["total_issues_found"] = issues
    report["data_quality_passed"] = issues == 0

    return report


def print_quality_report(df: pd.DataFrame) -> None:
    """Pretty-print the quality report to stdout."""
    r = check_data_quality(df)

    print("=" * 60)
    print("  DATA QUALITY REPORT")
    print("=" * 60)
    print(f"  Total issues found : {r['total_issues_found']}")
    print(f"  Quality check      : {'PASSED' if r['data_quality_passed'] else 'FAILED'}")

    print("\n-- Missing Values ------------------------------------------")
    if r["missing_values"]:
        for col, cnt in r["missing_values"].items():
            print(f"  {col:<25} {cnt:>8,}")
    else:
        print("  None.")

    print("\n-- Duplicate Rows ------------------------------------------")
    print(f"  {r['duplicate_rows']:,} duplicate rows detected.")

    print("\n-- Invalid Categorical Values ------------------------------")
    if r["invalid_categorical_values"]:
        for col, info in r["invalid_categorical_values"].items():
            print(f"  {col}: {info['count']:,} invalid  -> {info['values']}")
    else:
        print("  None.")

    print("\n-- Out-of-Range Numeric Values -----------------------------")
    if r["out_of_range_numerics"]:
        for col, info in r["out_of_range_numerics"].items():
            print(f"  {col}: {info['count']:,} rows  "
                  f"(actual min={info['min']}, max={info['max']}, "
                  f"bounds={info['bounds']})")
    else:
        print("  None.")

    print("\n-- Invalid Target Values -----------------------------------")
    print(f"  {r['invalid_target_values']:,} rows with target not in {{0, 1}}.")

    print("\n-- Negative Values in Non-Negative Columns -----------------")
    if r["negative_values"]:
        for col, cnt in r["negative_values"].items():
            print(f"  {col:<25} {cnt:>8,}")
    else:
        print("  None.")

    print("=" * 60)


# ── Cleaning pipeline ──────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning and preprocessing steps.

    Steps
    -----
    1. Drop exact duplicates (excluding LoanID)
    2. Drop rows with missing values (none expected, but defensive)
    3. Strip whitespace from string columns
    4. Encode binary Yes/No columns to int (1/0)
    5. Clip numeric features to valid domain bounds
    6. Drop the LoanID column (not a feature)
    7. Reset index

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from load_data().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for EDA and feature engineering.
    """
    df = df.copy()

    # 1. Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=df.columns.difference([ID_COLUMN]))
    dropped_dups = before - len(df)

    # 2. Drop rows with any missing values
    before = len(df)
    df = df.dropna()
    dropped_na = before - len(df)

    # 3. Strip whitespace from object columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 4. Encode binary Yes/No -> 1/0
    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).astype(int)

    # 5. Clip numerics to valid bounds
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # 6. Drop ID column
    df = df.drop(columns=[ID_COLUMN], errors="ignore")

    # 7. Reset index
    df = df.reset_index(drop=True)

    print(f"[clean_data] Duplicates removed : {dropped_dups:,}")
    print(f"[clean_data] Rows with NaN removed: {dropped_na:,}")
    print(f"[clean_data] Final shape         : {df.shape[0]:,} rows x {df.shape[1]} cols")

    return df
