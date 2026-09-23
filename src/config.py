"""
config.py
---------
Central configuration for the Loan Default Risk Analytics project.
All paths, constants, and model hyper-parameters are defined here so
every other module imports from one single source of truth.
"""

from pathlib import Path

# ── Project root (one level above this file) ──────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = ROOT_DIR / "data"
RAW_DATA_PATH   = DATA_DIR / "Loan_default.csv"

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR      = ROOT_DIR / "outputs"
FIGURES_DIR     = OUTPUT_DIR / "figures"
TABLES_DIR      = OUTPUT_DIR / "tables"

# ── Model artefact paths ──────────────────────────────────────────────────────
MODELS_DIR              = ROOT_DIR / "models"
MODEL_PATH              = MODELS_DIR / "random_forest_model.joblib"
PREPROCESSOR_PATH       = MODELS_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH      = MODELS_DIR / "feature_names.joblib"

# ── Dataset column definitions ────────────────────────────────────────────────
TARGET_COLUMN = "Default"
ID_COLUMN     = "LoanID"

NUMERIC_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
]

CATEGORICAL_FEATURES = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "LoanPurpose",
]

BINARY_FEATURES = [
    "HasMortgage",
    "HasDependents",
    "HasCoSigner",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# ── Random seed (for reproducibility) ────────────────────────────────────────
RANDOM_STATE = 42

# ── Train / test split ────────────────────────────────────────────────────────
TEST_SIZE = 0.20

# ── Random Forest hyper-parameters ───────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":  200,
    "max_depth":     None,
    "class_weight":  "balanced",
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}

# ── Classification threshold (for imbalanced data tuning) ────────────────────
DECISION_THRESHOLD = 0.40
