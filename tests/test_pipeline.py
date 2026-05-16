"""
test_pipeline.py
----------------
Unit tests for the ML data pipeline.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generator.generate_data          import generate_patient_data
from src.preprocessor.preprocess               import (
    remove_duplicates, filter_outliers, impute_missing_values
)
from src.feature_engineering.feature_engineer  import (
    compute_risk_score, flag_high_risk_patients,
    bin_age_groups, compute_procedure_intensity
)
from src.validation.validator                  import DataValidator


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    """Small raw dataset for testing."""
    return generate_patient_data(n=500)


@pytest.fixture
def clean_df():
    """Pre-cleaned DataFrame for feature engineering tests."""
    return pd.DataFrame({
        "patient_id":       ["PAT001", "PAT002", "PAT003"],
        "age":              [45, 72, 60],
        "n_diagnoses":      [2, 5, 3],
        "n_procedures":     [3, 8, 2],
        "length_of_stay":   [5, 12, 3],
        "prior_admissions": [0, 3, 1],
        "admission_date":   ["2023-06-15", "2023-11-20", "2024-01-05"],
        "primary_diagnosis":["E11", "I50", "J44"],
        "readmitted_30d":   [0, 1, 0],
    })


# ── Data Generation Tests ──────────────────────────────────────────

class TestDataGeneration:

    def test_correct_record_count(self, raw_df):
        """Should generate more than base n due to injected duplicates."""
        assert len(raw_df) > 500

    def test_required_columns_exist(self, raw_df):
        required = ["patient_id", "age", "length_of_stay", "readmitted_30d"]
        for col in required:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_target_variable_is_binary(self, raw_df):
        assert set(raw_df["readmitted_30d"].dropna().unique()).issubset({0, 1})

    def test_has_missing_values(self, raw_df):
        """Raw data should have injected nulls."""
        assert raw_df.isnull().sum().sum() > 0

    def test_has_duplicates(self, raw_df):
        """Raw data should have injected duplicates."""
        assert raw_df.duplicated(subset=["patient_id"]).sum() > 0


# ── Preprocessing Tests ────────────────────────────────────────────

class TestPreprocessing:

    def test_remove_duplicates(self, raw_df):
        deduped = remove_duplicates(raw_df)
        assert deduped.duplicated(subset=["patient_id"]).sum() == 0

    def test_filter_outliers_removes_extreme_ages(self, raw_df):
        filtered = filter_outliers(raw_df)
        assert filtered["age"].dropna().max() <= 110

    def test_filter_outliers_removes_extreme_los(self, raw_df):
        filtered = filter_outliers(raw_df)
        assert filtered["length_of_stay"].dropna().max() <= 60

    def test_impute_fills_nulls(self, raw_df):
        imputed = impute_missing_values(raw_df)
        for col in ["age", "n_procedures"]:
            assert imputed[col].isnull().sum() == 0


# ── Feature Engineering Tests ──────────────────────────────────────

class TestFeatureEngineering:

    def test_risk_score_is_positive(self, clean_df):
        result = compute_risk_score(clean_df)
        assert (result["readmission_risk_score"] >= 0).all()

    def test_high_risk_flag_is_binary(self, clean_df):
        result = flag_high_risk_patients(clean_df)
        assert set(result["high_risk_patient"].unique()).issubset({0, 1})

    def test_high_risk_flag_correct_logic(self, clean_df):
        """Patient aged 72 with 5 diagnoses should be flagged high risk."""
        result = flag_high_risk_patients(clean_df)
        assert result.loc[result["age"] == 72, "high_risk_patient"].values[0] == 1

    def test_age_group_bins_correctly(self, clean_df):
        result = bin_age_groups(clean_df)
        assert "age_group" in result.columns
        assert result.loc[result["age"] == 45, "age_group"].values[0] == "41-60"

    def test_procedure_intensity_no_division_by_zero(self, clean_df):
        clean_df.loc[0, "length_of_stay"] = 0
        result = compute_procedure_intensity(clean_df)
        assert result["procedure_intensity"].isnull().sum() == 0
        assert np.isinf(result["procedure_intensity"]).sum() == 0


# ── Validation Tests ───────────────────────────────────────────────

class TestValidator:

    def test_null_check_passes_on_clean_data(self, clean_df):
        v = DataValidator()
        v.check_no_nulls(clean_df, "Test")
        assert v.failed == 0

    def test_null_check_fails_on_dirty_data(self, clean_df):
        clean_df.loc[0, "age"] = np.nan
        v = DataValidator()
        v.check_no_nulls(clean_df, "Test")
        assert v.failed == 1

    def test_duplicate_check_passes(self, clean_df):
        v = DataValidator()
        v.check_no_duplicates(clean_df, "Test")
        assert v.failed == 0

    def test_row_count_check(self, clean_df):
        v = DataValidator()
        v.check_row_count(clean_df, "Test", min_rows=100)
        assert v.failed == 1  # only 3 rows, should fail
