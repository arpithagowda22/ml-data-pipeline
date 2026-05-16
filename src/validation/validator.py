"""
validator.py
------------
Runs automated data quality checks at each pipeline stage
and generates a validation report saved to outputs/.
"""

import os
import logging
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = "outputs/validation_report.txt"

# Expected schema for the final ML-ready dataset
EXPECTED_COLUMNS = [
    "patient_id", "age", "n_diagnoses", "n_procedures",
    "length_of_stay", "prior_admissions", "readmission_risk_score",
    "high_risk_patient", "procedure_intensity", "readmitted_30d"
]

NUMERICAL_RANGES = {
    "age":                    (18, 110),
    "length_of_stay":         (1, 60),
    "n_procedures":           (0, 30),
    "readmission_risk_score": (0, 200),
    "procedure_intensity":    (0, 30),
}


class DataValidator:
    """
    Validates DataFrames at each pipeline stage.
    Collects results and writes a final report.
    """

    def __init__(self):
        self.results = []
        self.passed  = 0
        self.failed  = 0

    def _record(self, check: str, passed: bool, detail: str = ""):
        status = "PASSED" if passed else "FAILED"
        self.results.append(f"  [{status}] {check}" + (f" — {detail}" if detail else ""))
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            logger.warning(f"Validation FAILED: {check} — {detail}")

    def check_no_nulls(self, df: pd.DataFrame, stage: str):
        """Check that no null values remain in the dataset."""
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        self._record(
            f"[{stage}] No null values",
            total_nulls == 0,
            f"{total_nulls} nulls found" if total_nulls > 0 else "clean"
        )

    def check_no_duplicates(self, df: pd.DataFrame, stage: str):
        """Check that no duplicate patient IDs exist."""
        dups = df.duplicated(subset=["patient_id"]).sum()
        self._record(
            f"[{stage}] No duplicate patient IDs",
            dups == 0,
            f"{dups} duplicates found" if dups > 0 else "clean"
        )

    def check_schema(self, df: pd.DataFrame, stage: str):
        """Check that expected columns are present."""
        missing = [c for c in EXPECTED_COLUMNS if c in df.columns or True]
        actually_missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        self._record(
            f"[{stage}] Schema validation",
            len(actually_missing) == 0,
            f"Missing columns: {actually_missing}" if actually_missing else "all expected columns present"
        )

    def check_numerical_ranges(self, df: pd.DataFrame, stage: str):
        """Check that numerical columns fall within expected ranges."""
        for col, (low, high) in NUMERICAL_RANGES.items():
            if col not in df.columns:
                continue
            out_of_range = (~df[col].between(low, high)).sum()
            self._record(
                f"[{stage}] {col} in range [{low}, {high}]",
                out_of_range == 0,
                f"{out_of_range} out-of-range values" if out_of_range > 0 else "ok"
            )

    def check_target_balance(self, df: pd.DataFrame, stage: str):
        """Check that target variable is not severely imbalanced (>95% one class)."""
        if "readmitted_30d" not in df.columns:
            return
        balance = df["readmitted_30d"].value_counts(normalize=True)
        majority_pct = balance.max() * 100
        self._record(
            f"[{stage}] Target class balance (majority < 95%)",
            majority_pct < 95,
            f"majority class = {majority_pct:.1f}%"
        )

    def check_row_count(self, df: pd.DataFrame, stage: str, min_rows: int = 5000):
        """Check that enough records remain after processing."""
        self._record(
            f"[{stage}] Sufficient rows (>= {min_rows:,})",
            len(df) >= min_rows,
            f"{len(df):,} rows"
        )

    def generate_report(self, stage_counts: dict) -> str:
        """Generate and save the full validation report."""
        os.makedirs("outputs", exist_ok=True)

        lines = [
            "=" * 60,
            "DATA QUALITY VALIDATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "Stage Record Counts:",
        ]
        for stage, count in stage_counts.items():
            lines.append(f"  {stage:<35}: {count:,}")

        lines += ["", "Validation Checks:", ""]
        lines += self.results
        lines += [
            "",
            "-" * 60,
            f"Total checks : {self.passed + self.failed}",
            f"Passed       : {self.passed}",
            f"Failed       : {self.failed}",
            f"Overall      : {'PASSED' if self.failed == 0 else 'FAILED'}",
            "=" * 60,
        ]

        report = "\n".join(lines)
        with open(OUTPUT_PATH, "w") as f:
            f.write(report)

        print("\n" + report)
        logger.info(f"Validation report saved to {OUTPUT_PATH}")
        return report


def run_validation(raw_df: pd.DataFrame, processed_df: pd.DataFrame, features_df: pd.DataFrame) -> DataValidator:
    """
    Run all validation checks across pipeline stages.

    Args:
        raw_df:       Raw generated data
        processed_df: Cleaned/preprocessed data
        features_df:  Final ML-ready feature dataset

    Returns:
        DataValidator instance with results
    """
    validator = DataValidator()

    # Processed data checks
    validator.check_no_duplicates(processed_df, "Preprocessing")
    validator.check_no_nulls(processed_df,      "Preprocessing")
    validator.check_row_count(processed_df,     "Preprocessing")

    # Feature data checks
    validator.check_no_nulls(features_df,          "Feature Engineering")
    validator.check_schema(features_df,            "Feature Engineering")
    validator.check_numerical_ranges(features_df,  "Feature Engineering")
    validator.check_target_balance(features_df,    "Feature Engineering")
    validator.check_row_count(features_df,         "Feature Engineering")

    stage_counts = {
        "Raw records generated":      len(raw_df),
        "After preprocessing":        len(processed_df),
        "After feature engineering":  len(features_df),
    }

    validator.generate_report(stage_counts)
    return validator


if __name__ == "__main__":
    raw  = pd.read_csv("data/raw/patient_data.csv")
    proc = pd.read_csv("data/processed/cleaned_data.csv")
    feat = pd.read_csv("data/features/ml_ready_dataset.csv")
    run_validation(raw, proc, feat)
