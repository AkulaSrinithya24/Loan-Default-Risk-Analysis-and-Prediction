"""
data_loader.py
--------------
Responsible for loading and performing an initial structural validation
of the raw loan default dataset.

Public API
----------
load_data(path)           -> pd.DataFrame
get_data_summary(df)      -> dict
print_data_summary(df)    -> None
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    ID_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
)


# ── Expected schema ────────────────────────────────────────────────────────────
EXPECTED_COLUMNS = (
    [ID_COLUMN]
    + NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + BINARY_FEATURES
    + [TARGET_COLUMN]
)


def load_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV file and return a validated DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.  Defaults to RAW_DATA_PATH from config.

    Returns
    -------
    pd.DataFrame
        Raw dataset with minimal dtype coercions applied.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If required columns are missing from the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    # ── Column validation ──────────────────────────────────────────────────────
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    # ── Lightweight dtype coercions ───────────────────────────────────────────
    # Ensure binary Yes/No columns are stored as plain strings
    for col in BINARY_FEATURES:
        df[col] = df[col].astype(str).str.strip().str.capitalize()

    # Ensure target is integer
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a structured summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys:
        shape, dtypes, missing_values, missing_pct,
        duplicate_rows, class_distribution, class_balance_pct,
        numeric_stats, categorical_stats
    """
    total_rows, total_cols = df.shape

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / total_rows * 100).round(2)

    # Duplicates (excluding ID column)
    duplicate_rows = df.drop(columns=[ID_COLUMN], errors="ignore").duplicated().sum()

    # Class distribution
    class_dist = df[TARGET_COLUMN].value_counts().sort_index()
    class_pct = (class_dist / total_rows * 100).round(2)

    # Numeric stats
    numeric_stats = df[NUMERIC_FEATURES].describe().round(2)

    # Categorical value counts
    categorical_stats = {
        col: df[col].value_counts(dropna=False).to_dict()
        for col in CATEGORICAL_FEATURES + BINARY_FEATURES
    }

    return {
        "shape":                (total_rows, total_cols),
        "dtypes":               df.dtypes.astype(str).to_dict(),
        "missing_values":       missing[missing > 0].to_dict(),
        "missing_pct":          missing_pct[missing_pct > 0].to_dict(),
        "duplicate_rows":       int(duplicate_rows),
        "class_distribution":   class_dist.to_dict(),
        "class_balance_pct":    class_pct.to_dict(),
        "numeric_stats":        numeric_stats,
        "categorical_stats":    categorical_stats,
    }


def print_data_summary(df: pd.DataFrame) -> None:
    """Pretty-print the data summary to stdout."""
    s = get_data_summary(df)
    rows, cols = s["shape"]

    print("=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Rows        : {rows:,}")
    print(f"  Columns     : {cols}")
    print(f"  Duplicates  : {s['duplicate_rows']:,}")

    print("\n-- Missing Values ------------------------------------------")
    if s["missing_values"]:
        for col, count in s["missing_values"].items():
            print(f"  {col:<25} {count:>8,}  ({s['missing_pct'][col]:.2f}%)")
    else:
        print("  No missing values detected.")

    print("\n-- Target Distribution -------------------------------------")
    for label, count in s["class_distribution"].items():
        pct = s["class_balance_pct"][label]
        tag = "Default" if label == 1 else "No Default"
        print(f"  {tag:<15} ({label}): {count:>8,}  ({pct:.2f}%)")

    print("\n-- Numeric Feature Statistics ------------------------------")
    print(s["numeric_stats"].to_string())

    print("\n-- Categorical Feature Distributions -----------------------")
    for col, counts in s["categorical_stats"].items():
        print(f"\n  {col}:")
        for val, cnt in counts.items():
            print(f"    {str(val):<20} {cnt:>8,}")

    print("=" * 60)
