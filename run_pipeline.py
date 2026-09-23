"""
run_pipeline.py
---------------
One-shot script that executes the full data → model pipeline:
  1. Load & validate raw data
  2. Clean & preprocess
  3. Feature engineering
  4. Train Random Forest model
  5. Evaluate and print metrics
  6. Save model artefacts to models/

Run from the project root:
    python run_pipeline.py
"""

import sys
from pathlib import Path

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.feature_engineering import build_preprocessor
from src.model import train_model, evaluate_model, save_artefacts
from src.config import RAW_DATA_PATH, TEST_SIZE, RANDOM_STATE

from sklearn.model_selection import train_test_split


def main():
    print("=" * 60)
    print("  Loan Default Risk Analytics — Full Pipeline")
    print("=" * 60)

    # 1. Load
    print("\n[1/5] Loading data …")
    df = load_data(RAW_DATA_PATH)
    print(f"      Loaded {len(df):,} rows × {df.shape[1]} columns.")

    # 2. Clean
    print("\n[2/5] Cleaning & preprocessing …")
    df_clean = clean_data(df)
    print(f"      Clean dataset: {len(df_clean):,} rows.")

    # 3. Feature engineering
    print("\n[3/5] Building feature pipeline …")
    X, y, preprocessor = build_preprocessor(df_clean)

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 5. Train
    print("\n[4/5] Training Random Forest model …")
    model = train_model(X_train, y_train, preprocessor)

    # 6. Evaluate
    print("\n[5/5] Evaluating model …")
    evaluate_model(model, X_test, y_test)

    # 7. Save
    save_artefacts(model, preprocessor)
    print("\nPipeline complete. Artefacts saved to models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
