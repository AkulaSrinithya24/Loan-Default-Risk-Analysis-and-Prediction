"""
feature_engineering.py
-----------------------
Builds a reusable scikit-learn preprocessing pipeline and prepares
X / y arrays ready for model training.

Public API
----------
build_preprocessor(df)   -> (X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer)
get_feature_names(preprocessor, df) -> list[str]
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

from src.config import (
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
)


# ── Column groups (post-clean: binary already 0/1 ints) ───────────────────────
NUMERIC_COLS     = NUMERIC_FEATURES          # 9 cols — scale
CATEGORICAL_COLS = CATEGORICAL_FEATURES      # 4 cols — ordinal encode
PASSTHROUGH_COLS = BINARY_FEATURES           # 3 cols — already numeric, pass through


def build_preprocessor(df: pd.DataFrame):
    """Build the preprocessing ColumnTransformer and return X, y, preprocessor.

    Steps
    -----
    - Numeric features  → StandardScaler
    - Categorical features → OrdinalEncoder (handle_unknown='use_encoded_value')
    - Binary int features → passthrough (already 0/1)

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame from clean_data() — must contain all feature columns
        and the TARGET_COLUMN.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all feature columns, no target).
    y : pd.Series
        Target vector (0 / 1).
    preprocessor : ColumnTransformer
        Unfitted preprocessor (fit on training data only to avoid leakage).
    """
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS + PASSTHROUGH_COLS

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_transformer,     NUMERIC_COLS),
            ("cat",  categorical_transformer, CATEGORICAL_COLS),
            ("pass", "passthrough",           PASSTHROUGH_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")

    return X, y, preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Return the ordered list of output feature names from the fitted preprocessor."""
    return list(preprocessor.get_feature_names_out())
