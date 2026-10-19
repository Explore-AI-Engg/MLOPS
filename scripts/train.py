"""Train, track, evaluate, and package the heart disease classifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from sklearn.model_selection import train_test_split

from app.config import ARTIFACTS_DIR, RAW_DATA_PATH
from app.data import load_raw_data, prepare_training_data
from app.modeling import RANDOM_STATE, candidate_models, evaluate_model

MIN_ROC_AUC = 0.70
EXPERIMENT_NAME = "heart-disease-classification"


def file_hash(path: Path) -> str:
    """Return a stable SHA-256 fingerprint for a dataset file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track_result(result, dataset_hash: str, output_dir: Path) -> None:
    """Log one model candidate and its artifacts to the active MLflow run."""
    mlflow.set_tag("dataset_sha256", dataset_hash)
    mlflow.set_tag("candidate", result.name)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("selection_metric", "cv_roc_auc")
    for metric, value in result.cv_metrics.items():
        mlflow.log_metric(f"cv_{metric}", value)
    for metric, value in result.holdout_metrics.items():
        mlflow.log_metric(f"holdout_{metric}", value)
    model_path = output_dir / f"{result.name}.joblib"
    joblib.dump(result.pipeline, model_path)
    mlflow.log_artifact(str(model_path), artifact_path="model")
    plot_path = output_dir / f"{result.name}_metrics.png"
    metric_names = list(result.cv_metrics)
    plt.figure(figsize=(8, 4))
    plt.bar(metric_names, [result.cv_metrics[name] for name in metric_names])
    plt.ylim(0, 1)
    plt.ylabel("Cross-validation score")
    plt.title(f"{result.name} cross-validation metrics")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    mlflow.log_artifact(str(plot_path), artifact_path="plots")


def train_all(
    input_path: Path = RAW_DATA_PATH,
    output_dir: Path = ARTIFACTS_DIR,
    tracking_uri: str = "file:./mlruns",
) -> dict:
    """Train both candidates, track them, and persist the champion."""
    raw = load_raw_data(input_path)
    features, target = prepare_training_data(raw)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    dataset_hash = file_hash(input_path)
    results = []
    for name, classifier in candidate_models().items():
        with mlflow.start_run(run_name=name) as run:
            result = evaluate_model(name, classifier, x_train, y_train, x_test, y_test)
            track_result(result, dataset_hash, output_dir)
            results.append((run.info.run_id, result))

    eligible = [item for item in results if item[1].cv_metrics["roc_auc"] >= MIN_ROC_AUC]
    if not eligible:
        raise RuntimeError(
            f"No model met the minimum CV ROC-AUC gate of {MIN_ROC_AUC:.2f}"
        )
    run_id, champion = max(
        eligible,
        key=lambda item: (item[1].cv_metrics["roc_auc"], item[1].name == "logistic_regression"),
    )
    champion_path = output_dir / "model.joblib"
    joblib.dump(champion.pipeline, champion_path)
    metadata = {
        "model_name": champion.name,
        "run_id": run_id,
        "dataset_sha256": dataset_hash,
        "selection_metric": "cv_roc_auc",
        "minimum_cv_roc_auc": MIN_ROC_AUC,
        "cv_metrics": champion.cv_metrics,
        "holdout_metrics": champion.holdout_metrics,
        "feature_count": len(features.columns),
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    args = parser.parse_args()
    metadata = train_all(args.input, args.output_dir, args.tracking_uri)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
