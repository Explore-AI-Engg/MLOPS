import pandas as pd
import pytest

from app.config import EXPECTED_COLUMNS, FEATURE_COLUMNS
from app.data import build_preprocessor, prepare_training_data, validate_schema


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
            [57, 0, 3, 130, 236, 0, 0, 174, 0, 0.0, 2, 1, 3, 2],
            [52, 1, 2, 172, 199, 1, 0, 162, 0, 0.5, 2, 0, 3, 4],
        ],
        columns=EXPECTED_COLUMNS,
    )


def test_target_is_binary_and_features_are_preserved() -> None:
    features, target = prepare_training_data(sample_frame())
    assert list(features.columns) == FEATURE_COLUMNS
    assert target.tolist() == [0, 1, 1]


def test_preprocessor_handles_missing_values_and_unknown_categories() -> None:
    frame = sample_frame()
    frame.loc[0, "chol"] = None
    features, target = prepare_training_data(frame)
    transformed = build_preprocessor().fit_transform(features, target)
    assert transformed.shape[0] == len(frame)
    assert transformed.shape[1] > len(FEATURE_COLUMNS)
    assert pd.isna(transformed).sum() == 0


def test_schema_rejects_missing_feature() -> None:
    frame = sample_frame().drop(columns=["thal"])
    with pytest.raises(ValueError, match="Expected columns"):
        validate_schema(frame)


def test_schema_rejects_missing_target() -> None:
    frame = sample_frame()
    frame.loc[0, "target"] = None
    with pytest.raises(ValueError, match="Target must not contain missing"):
        prepare_training_data(frame)
