"""Generate reproducible exploratory analysis outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from app.config import EDA_OUTPUT_DIR, FEATURE_COLUMNS, RAW_DATA_PATH, TARGET_COLUMN
from app.data import load_raw_data, prepare_training_data


def generate_eda(output_dir: Path = EDA_OUTPUT_DIR, input_path: Path = RAW_DATA_PATH) -> dict:
    """Write feature, correlation, and class-balance outputs."""
    raw = load_raw_data(input_path)
    features, target = prepare_training_data(raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    feature_axes = features.hist(figsize=(14, 10), bins=15, edgecolor="white")
    feature_axes[0, 0].figure.suptitle("Heart Disease Feature Distributions")
    feature_axes[0, 0].figure.tight_layout()
    feature_axes[0, 0].figure.savefig(output_dir / "feature_distributions.png", dpi=150)
    plt.close(feature_axes[0, 0].figure)

    correlation = pd.concat([features, target.rename(TARGET_COLUMN)], axis=1).corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    sns.heatmap(correlation, cmap="vlag", center=0, square=True)
    plt.title("Heart Disease Feature Correlations")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    counts = target.value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, hue=counts.index.astype(str), legend=False)
    plt.xlabel("Binary target (0=no disease, 1=disease)")
    plt.ylabel("Rows")
    plt.title("Class Balance")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=150)
    plt.close()

    summary = {
        "rows": len(raw),
        "features": FEATURE_COLUMNS,
        "missing_values": {key: int(value) for key, value in raw.isna().sum().items()},
        "class_counts": {str(key): int(value) for key, value in target.value_counts().sort_index().items()},
        "target_mapping": "0 means no disease; values greater than 0 mean disease present",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=EDA_OUTPUT_DIR)
    args = parser.parse_args()
    summary = generate_eda(args.output_dir, args.input)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
