"""Download and validate the UCI Cleveland heart-disease dataset."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

from app.config import (
    DATA_SOURCE_URL,
    EXPECTED_COLUMNS,
    RAW_DATA_PATH,
)

RAW_COLUMNS = EXPECTED_COLUMNS


def download_dataset(url: str = DATA_SOURCE_URL) -> tuple[pd.DataFrame, str]:
    """Download the source archive and return the validated Cleveland data."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        archive_bytes = response.content
    except requests.RequestException as exc:
        raise RuntimeError(f"Unable to download dataset from {url}: {exc}") from exc

    try:
        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            member = "processed.cleveland.data"
            if member not in archive.namelist():
                raise RuntimeError(f"Dataset archive does not contain {member}")
            with archive.open(member) as source:
                frame = pd.read_csv(source, header=None, names=RAW_COLUMNS, na_values="?")
    except (OSError, zipfile.BadZipFile, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Unable to parse the UCI dataset archive: {exc}") from exc

    validate_dataset(frame)
    return frame, url


def validate_dataset(frame: pd.DataFrame) -> None:
    """Validate the source schema and target convention."""
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}")
    if len(frame) < 200:
        raise ValueError(f"Dataset is unexpectedly small: {len(frame)} rows")
    if not frame["target"].dropna().isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Target must contain only UCI values from 0 through 4")
    if frame["target"].isna().any():
        raise ValueError("Target contains missing values")


def save_dataset(frame: pd.DataFrame, source_url: str, output_path: Path = RAW_DATA_PATH) -> None:
    """Write the raw normalized CSV and provenance metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    metadata = {
        "source_url": source_url,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(frame),
        "columns": list(frame.columns),
        "target_mapping": "0 means no disease; values 1 through 4 mean disease present",
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DATA_SOURCE_URL)
    parser.add_argument("--output", type=Path, default=RAW_DATA_PATH)
    args = parser.parse_args()
    frame, source_url = download_dataset(args.url)
    save_dataset(frame, source_url, args.output)
    print(f"Saved {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
