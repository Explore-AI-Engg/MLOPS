"""Smoke-test a running Heart Disease API."""

from __future__ import annotations

import argparse

import requests

VALID_PAYLOAD = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    health = requests.get(f"{base}/health", timeout=10)
    health.raise_for_status()
    prediction = requests.post(f"{base}/predict", json=VALID_PAYLOAD, timeout=10)
    prediction.raise_for_status()
    invalid = requests.post(f"{base}/predict", json={"age": 300}, timeout=10)
    if invalid.status_code != 422:
        raise RuntimeError(f"Expected invalid input status 422, got {invalid.status_code}")
    print({"health": health.json(), "prediction": prediction.json(), "invalid_status": invalid.status_code})


if __name__ == "__main__":
    main()
