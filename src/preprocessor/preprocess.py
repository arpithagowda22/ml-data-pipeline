"""
preprocess.py
-------------
Cleans and standardizes raw patient data.
Handles missing values, removes duplicates, filters outliers,
and encodes categorical variables for downstream feature engineering.
"""

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH       = "data/raw/patient_data.csv"
PROCESSED_PATH = "data/processed/cleaned_data.csv"

NUMERICAL_COLS   = ["age", "n_diagnoses", "n_procedures", "length_of_stay", "prior_admissions"]
CATEGORICAL_COLS = ["gender", "race", "insurance", "primary_diagnosis", "discharge_type"]

# Valid ranges for numerical columns — anything outside is an outlier
VALID_RANGES = {
    "age":            (18, 110),
    "length_of_stay": (1, 60),
    "n_procedures":   (0, 30),
    "n_diagnoses":    (1, 20),
    "prior_admissions": (0, 20),
}


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load raw CSV into a DataFrame."""
    df = pd.read_csv(path)
    logger.info(f"Loaded raw data | rows={len(df):,} | cols={len(df.columns)}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate patient records based on patient_id."""
    before = len(df)
    df     = df.drop_duplicates(subset=["patient_id"])
    logger.info(f"Duplicates removed | dropped={before - len(df):,} | remaining={len(df):,}")
    return df


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove records with values outside clinically valid ranges.

    Uses predefined VALID_RANGES rather than statistical methods
    because clinical domain knowledge is more reliable here.
    """
    before = len(df)
    for col, (low, high) in VALID_RANGES.items():
        if col in df.columns:
            df = df[df[col].isna() | df[col].between(low, high)]
    logger.info(f"Outliers removed | dropped={before - len(df):,} | remaining={len(df):,}")
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values.

    Strategy:
        - Numerical columns : median imputation (robust to skew)
        - Categorical columns: mode imputation (most frequent value)
    """
    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col]    = df[col].fillna(median_val)
            logger.info(f"Imputed {col} with median={median_val:.1f}")

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col]  = df[col].fillna(mode_val)
            logger.info(f"Imputed {col} with mode='{mode_val}'")

    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and data types."""
    # Ensure correct types
    df["age"]             = df["age"].astype(int)
    df["n_procedures"]    = df["n_procedures"].astype(int)
    df["n_diagnoses"]     = df["n_diagnoses"].astype(int)
    df["length_of_stay"]  = df["length_of_stay"].astype(int)
    df["prior_admissions"]= df["prior_admissions"].astype(int)
    df["admission_date"]  = pd.to_datetime(df["admission_date"])
    df["gender"]          = df["gender"].str.upper().str.strip()
    df["primary_diagnosis"] = df["primary_diagnosis"].str.strip()
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode categorical columns for ML compatibility."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
    return df


def run_preprocessing(input_path: str = RAW_PATH) -> pd.DataFrame:
    """
    Run full preprocessing pipeline: load -> deduplicate -> filter
    -> impute -> standardize -> encode -> save.

    Args:
        input_path: Path to raw CSV file

    Returns:
        Cleaned DataFrame
    """
    os.makedirs("data/processed", exist_ok=True)

    df = load_raw_data(input_path)
    df = remove_duplicates(df)
    df = filter_outliers(df)
    df = impute_missing_values(df)
    df = standardize_columns(df)
    df = encode_categoricals(df)

    df.to_csv(PROCESSED_PATH, index=False)
    logger.info(f"Preprocessing complete | final_rows={len(df):,} | saved to {PROCESSED_PATH}")
    return df


if __name__ == "__main__":
    run_preprocessing()
