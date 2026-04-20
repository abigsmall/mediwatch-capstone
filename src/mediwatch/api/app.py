from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException

from mediwatch.artifacts import load_model, read_manifest
from mediwatch.dataset import frame_from_payloads
from mediwatch.api.schemas import PredictionRequest, PredictionResponse


app = FastAPI(
    title="MediWatch API",
    description="Patient readmission inference service for the MediWatch capstone.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    payload_dict = payload.model_dump()
    try:
        model = load_model()
        manifest = read_manifest()
        features = frame_from_payloads([payload_dict])
        probability = float(model.predict_proba(features)[:, 1][0])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(
        prediction="readmitted" if probability >= 0.5 else "not_readmitted",
        probability=probability,
        model_version=manifest["model_version"],
        timestamp=datetime.now(timezone.utc),
        features=payload_dict,
    )


@app.post("/predict-batch")
def predict_batch(payloads: list[PredictionRequest]) -> list[dict]:
    if not payloads:
        raise HTTPException(status_code=400, detail="At least one payload is required.")
    try:
        model = load_model()
        manifest = read_manifest()
        features = frame_from_payloads([payload.model_dump() for payload in payloads])
        probabilities = model.predict_proba(features)[:, 1]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {exc}") from exc

    return [
        {
            "prediction": "readmitted" if probability >= 0.5 else "not_readmitted",
            "probability": float(probability),
            "model_version": manifest["model_version"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for probability in probabilities
    ]
