# ML Data Pipeline for Patient Readmission Prediction

A production-style data engineering pipeline that generates, preprocesses, and transforms healthcare data for machine learning workflows. The pipeline covers data generation, cleaning, feature engineering, validation, and model-ready output — all runnable locally without any cloud accounts or API keys.

---

## Background

Hospital readmissions within 30 days are a major cost driver in healthcare, often indicating gaps in patient care or discharge planning. This pipeline simulates the data engineering work required to prepare patient data for a predictive ML model — the kind of work done daily in healthcare data teams.

This project focuses on the data engineering side, not the modeling side: getting raw, messy patient data into a clean, validated, feature-rich dataset that can be handed off directly to a data scientist or ML model.

---

## Architecture

```
Data Generator --> Raw CSV --> Preprocessing --> Cleaned Data --> Feature Engineering --> ML-Ready Dataset --> Validation Report
```

---

## Tech Stack

- Python 3.8+
- Pandas
- NumPy
- Scikit-learn (preprocessing utilities)
- Pytest (data validation tests)

---

## Project Structure

```
ml_pipeline/
├── src/
│   ├── data_generator/
│   │   └── generate_data.py        # Generates realistic synthetic patient data
│   ├── preprocessor/
│   │   └── preprocess.py           # Cleans and standardizes raw data
│   ├── feature_engineering/
│   │   └── feature_engineer.py     # Builds ML-ready features
│   └── validation/
│       └── validator.py            # Validates data quality at each stage
├── data/
│   ├── raw/                        # Raw generated data
│   ├── processed/                  # Cleaned data
│   └── features/                   # Final ML-ready dataset
├── outputs/
│   └── validation_report.txt       # Data quality report
├── tests/
│   └── test_pipeline.py            # Unit tests
├── run_pipeline.py                 # Single entry point to run everything
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/arpithagowda22/ml-data-pipeline.git
cd ml-data-pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python run_pipeline.py
```

No API keys, no cloud accounts, no configuration needed.

---

## Pipeline Stages

### Stage 1 - Data Generation
Generates 10,000 realistic synthetic patient records with intentional noise, missing values, and outliers mimicking real-world healthcare data quality issues.

Fields include patient demographics, admission details, diagnosis codes, length of stay, number of procedures, prior admissions, discharge disposition, and 30-day readmission label.

### Stage 2 - Preprocessing
- Drops duplicate patient records
- Handles missing values using median imputation (numerical) and mode imputation (categorical)
- Standardizes column names and data types
- Removes outliers using IQR-based filtering
- Encodes categorical variables

### Stage 3 - Feature Engineering
- Computes readmission_risk_score from prior admissions and length of stay
- Creates high_risk_patient flag based on age and comorbidity count
- Bins age into clinical categories (18-40, 41-60, 61-80, 80+)
- Extracts admission month and day-of-week from admission date
- Calculates procedure_intensity ratio

### Stage 4 - Validation
Runs automated data quality checks at each stage and produces a validation report covering row counts, null value checks, schema validation, range checks on numerical columns, and class balance of the target variable.

---

## Sample Output

```
Pipeline run complete.

Stage Summary:
  Raw records generated      : 10,000
  After preprocessing        : 9,743  (removed 257 duplicates/outliers)
  After feature engineering  : 9,743  (added 6 new features)

Target Variable Distribution:
  Readmitted (1)             : 2,631  (27.0%)
  Not readmitted (0)         : 7,112  (73.0%)

Data Quality:
  Null values in final dataset: 0
  Schema validation           : PASSED
  Range checks                : PASSED

Output saved to: data/features/ml_ready_dataset.csv
Validation report saved to: outputs/validation_report.txt
```

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

---

## Why This Matters

In healthcare data engineering, data quality directly impacts patient outcomes. A model trained on poorly preprocessed data can misclassify high-risk patients, leading to missed interventions. This pipeline enforces strict validation at every stage to ensure the ML team receives trustworthy, consistent data.

---

## Author

Arpitha Raghu - Data Engineer
LinkedIn: https://www.linkedin.com/in/arpitha2205/
GitHub: https://github.com/arpithagowda22
Email: arpithagowda2205@gmail.com
