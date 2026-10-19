"""Deterministic model construction and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from app.data import build_preprocessor

RANDOM_STATE = 42
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


@dataclass
class EvaluationResult:
    """Cross-validation and holdout metrics for one candidate."""

    name: str
    pipeline: Pipeline
    cv_metrics: dict[str, float]
    holdout_metrics: dict[str, float]


def candidate_models() -> dict[str, Any]:
    """Return the deterministic candidate classifier definitions."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def build_pipeline(classifier: Any) -> Pipeline:
    """Compose preprocessing and classifier into one persisted pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def evaluate_model(
    name: str,
    classifier: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationResult:
    """Fit and evaluate a candidate using CV and a final holdout."""
    pipeline = build_pipeline(classifier)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_result = cross_validate(
        pipeline,
        x_train,
        y_train,
        cv=folds,
        scoring=SCORING,
        return_train_score=False,
        error_score="raise",
    )
    cv_metrics = {
        metric: float(cv_result[f"test_{metric}"].mean()) for metric in SCORING
    }
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    holdout_metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    return EvaluationResult(name, pipeline, cv_metrics, holdout_metrics)
