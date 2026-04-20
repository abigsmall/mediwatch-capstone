from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class PredictionResponse(BaseModel):
    prediction: str = Field(description="Predicted readmission label.")
    probability: float = Field(description="Probability of readmission.")
    model_version: str
    timestamp: datetime
    features: dict[str, Any]
