"""
feature_engineer.py
-------------------
Builds ML-ready features from cleaned patient data.
Creates domain-specific features based on healthcare knowledge
to improve downstream model performance.
"""

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_PATH = "data/processed/cleaned_data.csv"
FEATURES_PATH  = "data/features/ml_ready_dataset.csv"

# High-risk diagnosis codes based on clinical literature
HIGH_RISK_DIAGNOSES = ["I50", "N18", "I21"]  # Heart Failure, CKD, Acute MI


def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite readmission risk score.

    Formula based on clinical risk factors:
        - Prior admissions carry the most weight (strongest predictor)
        - Length of stay indicates severity
        - Age adds baseline risk

    Higher score = higher readmission risk.
    """
    df["readmission_risk_score"] = (
        (df["prior_admissions"] * 3)
        + (df["length_of_stay"] * 0.5)
        + (df["age"] * 0.1)
        + (df["n_diagnoses"] * 0.5)
    ).round(2)

    logger.info("Feature created: readmission_risk_score")
    return df


def flag_high_risk_patients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag patients as high risk based on age and comorbidity count.

    Criteria: age >= 65 AND n_diagnoses >= 3
    This reflects CMS high-risk patient definitions commonly
    used in healthcare analytics.
    """
    df["high_risk_patient"] = (
        (df["age"] >= 65) & (df["n_diagnoses"] >= 3)
    ).astype(int)

    high_risk_count = df["high_risk_patient"].sum()
    logger.info(f"Feature created: high_risk_patient | flagged={high_risk_count:,}")
    return df


def bin_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin age into clinical age groups.

    Groups align with standard healthcare risk stratification:
        18-40  : Young adult
        41-60  : Middle aged
        61-80  : Older adult
        80+    : Elderly (highest readmission risk group)
    """
    bins   = [0, 40, 60, 80, 200]
    labels = ["18-40", "41-60", "61-80", "80+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    df["age_group"] = df["age_group"].astype(str)
    logger.info("Feature created: age_group")
    return df


def extract_admission_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from admission date.

    Admission timing can correlate with readmission risk —
    weekend admissions often have fewer resources available.
    """
    df["admission_month"]    = pd.to_datetime(df["admission_date"]).dt.month
    df["admission_dayofweek"]= pd.to_datetime(df["admission_date"]).dt.dayofweek
    df["is_weekend_admission"]= (df["admission_dayofweek"] >= 5).astype(int)
    logger.info("Features created: admission_month, admission_dayofweek, is_weekend_admission")
    return df


def compute_procedure_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute procedure intensity as procedures per day of stay.

    High intensity may indicate complex cases with higher readmission risk.
    """
    df["procedure_intensity"] = (
        df["n_procedures"] / df["length_of_stay"].replace(0, 1)
    ).round(3)
    logger.info("Feature created: procedure_intensity")
    return df


def flag_high_risk_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """Flag records with high-risk primary diagnosis codes."""
    df["high_risk_diagnosis"] = df["primary_diagnosis"].isin(HIGH_RISK_DIAGNOSES).astype(int)
    logger.info(f"Feature created: high_risk_diagnosis")
    return df


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select final feature set for ML model input."""
    feature_cols = [
        # Identifiers
        "patient_id",
        # Original features
        "age", "gender_encoded", "race_encoded", "insurance_encoded",
        "n_diagnoses", "n_procedures", "length_of_stay", "prior_admissions",
        "primary_diagnosis_encoded", "discharge_type_encoded",
        # Engineered features
        "readmission_risk_score", "high_risk_patient", "age_group",
        "admission_month", "admission_dayofweek", "is_weekend_admission",
        "procedure_intensity", "high_risk_diagnosis",
        # Target variable
        "readmitted_30d"
    ]
    available = [c for c in feature_cols if c in df.columns]
    return df[available]


def run_feature_engineering(input_path: str = PROCESSED_PATH) -> pd.DataFrame:
    """
    Run full feature engineering pipeline.

    Args:
        input_path: Path to cleaned/processed CSV

    Returns:
        ML-ready DataFrame with engineered features
    """
    os.makedirs("data/features", exist_ok=True)

    df = pd.read_csv(input_path)
    logger.info(f"Loaded processed data | rows={len(df):,}")

    df = compute_risk_score(df)
    df = flag_high_risk_patients(df)
    df = bin_age_groups(df)
    df = extract_admission_features(df)
    df = compute_procedure_intensity(df)
    df = flag_high_risk_diagnosis(df)
    df = select_final_features(df)

    df.to_csv(FEATURES_PATH, index=False)
    logger.info(f"Feature engineering complete | features={len(df.columns)} | rows={len(df):,} | saved to {FEATURES_PATH}")
    return df


if __name__ == "__main__":
    run_feature_engineering()
