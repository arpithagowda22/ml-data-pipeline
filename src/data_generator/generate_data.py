"""
generate_data.py
----------------
Generates realistic synthetic patient data for the readmission prediction pipeline.
Intentionally includes missing values, outliers, and duplicates to simulate
real-world healthcare data quality issues.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_RECORDS   = 10000
OUTPUT_PATH = "data/raw/patient_data.csv"

DIAGNOSIS_CODES = ["E11", "I50", "J44", "N18", "I10", "J18", "K92", "I21"]
DISCHARGE_TYPES = ["Home", "Home with Home Health", "Skilled Nursing Facility",
                   "Rehabilitation", "Against Medical Advice", "Expired"]
INSURANCE_TYPES = ["Medicare", "Medicaid", "Private", "Uninsured"]


def random_dates(start: str, end: str, n: int) -> pd.Series:
    """Generate n random dates between start and end."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    delta    = (end_dt - start_dt).days
    return pd.Series([
        (start_dt + timedelta(days=int(np.random.randint(0, delta)))).strftime("%Y-%m-%d")
        for _ in range(n)
    ])


def generate_patient_data(n: int = N_RECORDS) -> pd.DataFrame:
    """
    Generate synthetic patient records with realistic readmission patterns.

    Intentional data quality issues:
        - ~5% missing values in key columns
        - ~2% duplicate records
        - ~3% outlier values in numerical columns

    Args:
        n: Number of records to generate

    Returns:
        Raw patient DataFrame
    """
    ages             = np.random.randint(18, 95, n)
    prior_admissions = np.random.poisson(1.5, n).clip(0, 10)
    length_of_stay   = np.random.exponential(5, n).clip(1, 30).astype(int)
    n_procedures     = np.random.randint(0, 15, n)
    n_diagnoses      = np.random.randint(1, 10, n)

    # Readmission probability increases with age, prior admissions, length of stay
    readmission_prob = (
        0.1
        + 0.003 * (ages - 18)
        + 0.05  * prior_admissions
        + 0.01  * length_of_stay
    ).clip(0, 0.9)
    readmitted = (np.random.rand(n) < readmission_prob).astype(int)

    df = pd.DataFrame({
        "patient_id":        [f"PAT{str(i).zfill(6)}" for i in range(n)],
        "age":               ages,
        "gender":            np.random.choice(["M", "F"], n),
        "race":              np.random.choice(["White", "Black", "Hispanic", "Asian", "Other"], n),
        "insurance":         np.random.choice(INSURANCE_TYPES, n, p=[0.45, 0.25, 0.25, 0.05]),
        "admission_date":    random_dates("2022-01-01", "2024-12-31", n),
        "primary_diagnosis": np.random.choice(DIAGNOSIS_CODES, n),
        "n_diagnoses":       n_diagnoses,
        "n_procedures":      n_procedures,
        "length_of_stay":    length_of_stay,
        "prior_admissions":  prior_admissions,
        "discharge_type":    np.random.choice(DISCHARGE_TYPES, n),
        "readmitted_30d":    readmitted,
    })

    # Inject missing values (~5%)
    for col in ["age", "insurance", "n_procedures", "discharge_type"]:
        null_idx = np.random.choice(df.index, size=int(n * 0.05), replace=False)
        df.loc[null_idx, col] = np.nan

    # Inject outliers (~3%)
    outlier_idx = np.random.choice(df.index, size=int(n * 0.03), replace=False)
    df.loc[outlier_idx, "length_of_stay"] = np.random.randint(100, 500, len(outlier_idx))
    df.loc[outlier_idx, "age"]            = np.random.randint(150, 200, len(outlier_idx))

    # Inject duplicate records (~2%)
    dup_idx = np.random.choice(df.index, size=int(n * 0.02), replace=False)
    df      = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df


def run_data_generation() -> pd.DataFrame:
    """Generate and save raw patient data to CSV."""
    os.makedirs("data/raw", exist_ok=True)
    print("Generating synthetic patient data...")
    df = generate_patient_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Records generated : {len(df):,}")
    print(f"  Saved to          : {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    run_data_generation()
