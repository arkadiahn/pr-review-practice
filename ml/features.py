"""
Feature engineering for Spotify Top 100 popularity prediction.
Takes raw track data and produces a feature matrix ready for training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# BUG 1 (ML — Data Leakage, Medium):
# "popularity" is the target variable. Including it here means the model
# trains on the answer itself. Should be excluded from AUDIO_FEATURES.
ALL_FEATURES = AUDIO_FEATURES + ["popularity", "explicit", "key", "mode", "time_signature"]


def build_feature_matrix(df: pd.DataFrame, fit_scaler: bool = True) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Build a scaled feature matrix from raw track data.

    Args:
        df: Raw Spotify track DataFrame.
        fit_scaler: If True, fit a new scaler (training). If False, use a pre-fitted one.

    Returns:
        Tuple of (feature DataFrame, fitted StandardScaler).
    """
    df = df.copy()

    # Derived features
    df["energy_danceability_ratio"] = df["energy"] / (df["danceability"] + 1e-6)
    df["duration_minutes"] = df["duration_ms"] / 60_000
    df["explicit_int"] = df["explicit"].astype(int)

    feature_cols = ALL_FEATURES + ["energy_danceability_ratio", "duration_minutes", "explicit_int"]
    # Drop original columns that were transformed
    feature_cols = [c for c in feature_cols if c not in ("duration_ms", "explicit")]

    X = df[feature_cols].copy()

    scaler = StandardScaler()
    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), scaler


def split_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split dataframe into train/test sets.
    Stratify is not applicable for regression; simple random split.
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_size, random_state=random_state)
