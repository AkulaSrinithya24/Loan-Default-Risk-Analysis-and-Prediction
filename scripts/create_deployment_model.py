"""
scripts/create_deployment_model.py
-----------------------------------
Creates a lightweight Random Forest model suitable for Streamlit Community
Cloud deployment (where the full 706 MB model cannot be committed to GitHub).

What this script does
---------------------
- Uses the SAME dataset, feature columns, preprocessing pipeline,
  random_state (42), and decision threshold (0.40) as the production model.
- Trains a Random Forest with only 30 trees and max_depth=12 so the
  serialised file stays well under GitHub's 100 MB file size limit.
- Saves the pipeline to  models/deployment_model.joblib
- Does NOT overwrite models/random_forest_model.joblib (the full model).

Usage
-----
    python scripts/create_deployment_model.py

Requirements
------------
Run from the project root, or from any directory — the script resolves
PROJECT_ROOT automatically via __file__.
"""

import sys
from pathlib import Path

# ── Resolve project root regardless of where the script is called from ─────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    RAW_DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    DECISION_THRESHOLD,
    MODELS_DIR,
    DEPLOYMENT_MODEL_PATH,
)
from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.feature_engineering import build_preprocessor, get_feature_names

# ── Deployment model hyper-parameters ──────────────────────────────────────
# 30 trees keeps the file small (~10-20 MB) while preserving acceptable
# discrimination.  max_depth=12 caps tree size for further compression.
DEPLOY_RF_PARAMS = {
    "n_estimators":  30,
    "max_depth":     12,
    "class_weight":  "balanced",   # same imbalance handling as full model
    "random_state":  RANDOM_STATE, # same seed → reproducible
    "n_jobs":        -1,
}


def main() -> None:
    print("=" * 60)
    print("  Deployment Model Creator")
    print("  Target: models/deployment_model.joblib")
    print("=" * 60)

    # 1. Load and clean data (same pipeline as full model) ─────────────────
    print("\n[1/5] Loading and cleaning dataset …")
    df = clean_data(load_data(RAW_DATA_PATH))
    print(f"      Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # 2. Build feature matrix and preprocessor ────────────────────────────
    print("[2/5] Building feature matrix and preprocessor …")
    X, y, preprocessor = build_preprocessor(df)
    print(f"      Features: {X.columns.tolist()}")

    # 3. Stratified train/test split (same split as full model) ───────────
    print("[3/5] Splitting train / test …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")
    print(f"      Train default rate: {y_train.mean()*100:.2f}%")
    print(f"      Test  default rate: {y_test.mean()*100:.2f}%")

    # 4. Train lightweight Random Forest ──────────────────────────────────
    print(f"[4/5] Training deployment Random Forest …")
    print(f"      Params: {DEPLOY_RF_PARAMS}")
    clf = RandomForestClassifier(**DEPLOY_RF_PARAMS)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])
    pipeline.fit(X_train, y_train)
    print("      Training complete.")

    # 5. Quick evaluation at the project's decision threshold ─────────────
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, accuracy_score,
    )
    import numpy as np

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    print(f"\n  Evaluation on test set (threshold = {DECISION_THRESHOLD}):")
    print(f"  ROC-AUC       : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  Avg Precision : {average_precision_score(y_test, y_prob):.4f}")
    print(f"  F1 (Default)  : {f1_score(y_test, y_pred):.4f}")
    print(f"  Accuracy      : {accuracy_score(y_test, y_pred):.4f}")

    # 6. Save ─────────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving deployment model …")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, DEPLOYMENT_MODEL_PATH, compress=3)

    size_mb = DEPLOYMENT_MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"      Saved  : {DEPLOYMENT_MODEL_PATH}")
    print(f"      Size   : {size_mb:.1f} MB")

    # Sanity check: original model still untouched ─────────────────────
    from src.config import MODEL_PATH
    if MODEL_PATH.exists():
        orig_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"\n  Original model intact : {MODEL_PATH} ({orig_mb:.0f} MB)")

    print("\nDone. Commit models/deployment_model.joblib to GitHub.")
    print("=" * 60)


if __name__ == "__main__":
    main()
