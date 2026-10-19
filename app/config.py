from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "heart.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "heart_processed.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EDA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

DATA_SOURCE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

TARGET_COLUMN = "target"
EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]