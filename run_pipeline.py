"""
run_pipeline.py
---------------
Single entry point to run the full ML data pipeline.

Usage:
    python run_pipeline.py

Stages:
    1. Generate synthetic patient data
    2. Preprocess and clean raw data
    3. Engineer ML-ready features
    4. Validate data quality and generate report
"""

import sys
import time
from src.data_generator.generate_data       import run_data_generation
from src.preprocessor.preprocess            import run_preprocessing
from src.feature_engineering.feature_engineer import run_feature_engineering
from src.validation.validator               import run_validation


def run_pipeline():
    print("\n" + "=" * 60)
    print("ML DATA PIPELINE — Patient Readmission Prediction")
    print("=" * 60 + "\n")

    start = time.time()

    # Stage 1 — Generate data
    print("Stage 1/4 — Generating synthetic patient data...")
    raw_df = run_data_generation()
    print()

    # Stage 2 — Preprocess
    print("Stage 2/4 — Preprocessing and cleaning...")
    processed_df = run_preprocessing()
    print()

    # Stage 3 — Feature engineering
    print("Stage 3/4 — Engineering features...")
    features_df = run_feature_engineering()
    print()

    # Stage 4 — Validate
    print("Stage 4/4 — Running data quality validation...")
    validator = run_validation(raw_df, processed_df, features_df)
    print()

    elapsed = round(time.time() - start, 2)

    print("=" * 60)
    print(f"Pipeline complete in {elapsed}s")
    print(f"ML-ready dataset  : data/features/ml_ready_dataset.csv")
    print(f"Validation report : outputs/validation_report.txt")
    print("=" * 60 + "\n")

    if validator.failed > 0:
        print(f"WARNING: {validator.failed} validation check(s) failed. Review the report.")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
