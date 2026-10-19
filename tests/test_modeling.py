from pathlib import Path

import joblib

from app.data import prepare_training_data
from app.modeling import RANDOM_STATE, build_pipeline, candidate_models
from tests.test_data import sample_frame


def test_candidate_models_are_deterministic() -> None:
    assert set(candidate_models()) == {"logistic_regression", "random_forest"}
    assert candidate_models()["random_forest"].random_state == RANDOM_STATE


def test_persisted_pipeline_reproduces_predictions(tmp_path: Path) -> None:
    features, target = prepare_training_data(sample_frame())
    pipeline = build_pipeline(candidate_models()["logistic_regression"])
    pipeline.fit(features, target)
    before = pipeline.predict(features).tolist()
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    reloaded = joblib.load(path)
    assert reloaded.predict(features).tolist() == before
    assert len(reloaded.predict_proba(features)) == len(features)
