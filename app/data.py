"""Data validation and reproducible preprocessing for heart disease models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import (
    CATEGORICAL_FEATURES,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load and validate the downloaded source CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run scripts/download_data.py first."
        )
    frame = pd.read_csv(path)
    validate_schema(frame, require_target=True)
    return frame


def validate_schema(frame: pd.DataFrame, require_target: bool = True) -> None:
    """Validate feature names and target presence before transformation."""
    expected = EXPECTED_COLUMNS if require_target else FEATURE_COLUMNS
    actual = list(frame.columns)
    if actual != expected:
        raise ValueError(f"Expected columns {expected}, received {actual}")
    if frame.empty:
        raise ValueError("Dataset must contain at least one row")
    if require_target and frame[TARGET_COLUMN].isna().any():
        raise ValueError("Target must not contain missing values")


def prepare_training_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate raw data and convert the UCI multi-class target to binary."""
    validate_schema(frame, require_target=True)
    features = frame[FEATURE_COLUMNS].copy()
    target = (pd.to_numeric(frame[TARGET_COLUMN]) > 0).astype(int)
    return features, target


def build_preprocessor() -> ColumnTransformer:
    """Build the shared numeric and categorical preprocessing transformer."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def save_processed_data(frame: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    """Normalize the target and save a validated processed CSV for inspection."""
    features, target = prepare_training_data(frame)
    processed = features.copy()
    processed[TARGET_COLUMN] = target
    path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(path, index=False)
