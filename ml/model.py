"""
Model training and serialization for Spotify popularity prediction.
Uses a RandomForestRegressor to predict track popularity (0–100).
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score


MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor on the provided feature matrix.

    Args:
        X_train: Feature matrix (already scaled).
        y_train: Target popularity scores.

    Returns:
        Fitted RandomForestRegressor.
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the model and return a metrics dictionary.

    Cross-validation uses neg_mean_absolute_error so higher = better,
    consistent with sklearn's scoring convention.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # BUG 2 (ML — Wrong metric direction, Easy):
    # cross_val_score with scoring="r2" returns R² values (higher is better).
    # The code then takes the *minimum* CV score as the "best" score, which is
    # the worst fold. Should be np.mean(cv_scores) or max, not min.
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring="r2")
    best_cv_score = np.min(cv_scores)

    return {
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "cv_r2_best": round(best_cv_score, 4),
        "cv_r2_scores": cv_scores.tolist(),
    }


def save_model(model: RandomForestRegressor, path: str = MODEL_PATH) -> None:
    """Serialize model to disk with pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str = MODEL_PATH) -> RandomForestRegressor:
    """Load a serialized model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
