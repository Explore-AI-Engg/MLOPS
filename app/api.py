"""FastAPI inference service with logging and Prometheus metrics."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from app.config import ARTIFACTS_DIR, FEATURE_COLUMNS

logger = logging.getLogger("heart-disease-api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

REQUEST_COUNT = Counter(
    "heart_disease_api_requests_total",
    "Number of API requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "heart_disease_api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
)


class PredictionRequest(BaseModel):
    """Validated feature payload for one prediction."""

    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=0, le=120)
    sex: float = Field(ge=0, le=1)
    cp: float = Field(ge=0, le=3)
    trestbps: float = Field(ge=0, le=300)
    chol: float = Field(ge=0, le=700)
    fbs: float = Field(ge=0, le=1)
    restecg: float = Field(ge=0, le=2)
    thalach: float = Field(ge=0, le=300)
    exang: float = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-10, le=20)
    slope: float = Field(ge=0, le=3)
    ca: float = Field(ge=0, le=3)
    thal: float = Field(ge=0, le=7)


class ModelService:
    """Load and invoke the persisted pipeline."""

    def __init__(self, model_path: Path = ARTIFACTS_DIR / "model.joblib") -> None:
        self.model_path = model_path
        self.model = None
        self.model_version = "unavailable"
        metadata_path = model_path.with_name("model_metadata.json")
        if model_path.exists():
            self.model = joblib.load(model_path)
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.model_version = metadata.get("run_id", metadata.get("model_name", "unknown"))

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, payload: PredictionRequest) -> tuple[int, float]:
        if not self.ready:
            raise RuntimeError("Model artifact is not available")
        frame = pd.DataFrame([payload.model_dump()], columns=FEATURE_COLUMNS)
        prediction = int(self.model.predict(frame)[0])
        confidence = float(self.model.predict_proba(frame)[0][prediction])
        return prediction, confidence


service = ModelService(Path(os.getenv("MODEL_PATH", str(ARTIFACTS_DIR / "model.joblib"))))
app = FastAPI(title="Heart Disease Classifier", version="1.0.0")


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    started = time.perf_counter()
    status = "error"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        duration = time.perf_counter() - started
        endpoint = request.url.path
        REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        logger.info(
            json.dumps(
                {
                    "event": "api_request",
                    "method": request.method,
                    "path": endpoint,
                    "status": status,
                    "latency_ms": round(duration * 1000, 3),
                }
            )
        )


@app.get("/health")
def health() -> JSONResponse:
    if not service.ready:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "model": service.model_version},
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ready", "model": service.model_version},
    )


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    prediction, confidence = service.predict(payload)
    return {
        "prediction": prediction,
        "confidence": round(confidence, 6),
        "model_version": service.model_version,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
