"""
FastAPI serving layer for the Spotify popularity prediction model.
Exposes /predict and /health endpoints.
"""

import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.model import load_model
from ml.features import build_feature_matrix, AUDIO_FEATURES


app = FastAPI(
    title="Spotify Popularity Predictor",
    description="Predicts track popularity (0–100) from audio features.",
    version="0.1.0",
)

# Model is loaded once at startup
_model = None


def get_model():
    global _model
    if _model is None:
        model_path = os.environ.get("MODEL_PATH", "model.pkl")
        _model = load_model(model_path)
    return _model


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TrackFeatures(BaseModel):
    danceability: float = Field(..., ge=0.0, le=1.0)
    energy: float = Field(..., ge=0.0, le=1.0)
    loudness: float = Field(..., le=0.0, description="Typically negative dB value")
    speechiness: float = Field(..., ge=0.0, le=1.0)
    acousticness: float = Field(..., ge=0.0, le=1.0)
    instrumentalness: float = Field(..., ge=0.0, le=1.0)
    liveness: float = Field(..., ge=0.0, le=1.0)
    valence: float = Field(..., ge=0.0, le=1.0)
    tempo: float = Field(..., gt=0.0)
    duration_ms: int = Field(..., gt=0)
    explicit: bool = False
    key: int = Field(default=0, ge=0, le=11)
    mode: int = Field(default=1, ge=0, le=1)
    time_signature: int = Field(default=4, ge=1, le=7)


class PredictionResponse(BaseModel):
    predicted_popularity: float
    model_version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(track: TrackFeatures):
    """
    Predict the popularity of a Spotify track from its audio features.

    BUG 5 (Integration — Missing preprocessing at inference, Medium):
    The raw input values are fed directly to model.predict() without going
    through build_feature_matrix(). This means:
      - The derived features (energy_danceability_ratio, duration_minutes,
        explicit_int) are never computed.
      - The StandardScaler fitted during training is never applied.
    At inference time, build_feature_matrix(df, fit_scaler=False) should be
    called with the saved scaler, and the same feature column order must be used.
    """
    model = get_model()

    input_data = track.model_dump()
    df = pd.DataFrame([input_data])

    # BUG IS HERE: raw df passed directly — no scaler, no derived features
    feature_cols = AUDIO_FEATURES + ["explicit", "key", "mode", "time_signature"]
    X = df[feature_cols].values

    try:
        prediction = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Clamp to valid popularity range
    prediction = float(np.clip(prediction, 0, 100))

    model_version = os.environ.get("MODEL_VERSION", "unknown")
    return PredictionResponse(
        predicted_popularity=round(prediction, 2),
        model_version=model_version,
    )
