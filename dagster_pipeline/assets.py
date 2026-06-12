"""
Dagster asset definitions for the Spotify Top 100 ML pipeline.

Pipeline stages:
  raw_tracks → featured_tracks → trained_model → model_metrics
"""

import json
import os
import pandas as pd
from dagster import (
    asset,
    AssetExecutionContext,
    MetadataValue,
    Output,
)

from dagster_pipeline.partitions import monthly_partitions
from dagster_pipeline.resources import LakeFSResource, ModelRegistryResource
from ml.features import build_feature_matrix, split_train_test
from ml.model import train_model, evaluate_model, save_model


DATA_DIR = os.environ.get("DATA_DIR", "data/")


# ---------------------------------------------------------------------------
# Asset 1: raw_tracks
# ---------------------------------------------------------------------------

@asset(partitions_def=monthly_partitions)
def raw_tracks(context: AssetExecutionContext) -> Output[pd.DataFrame]:
    """
    Load raw Spotify track data for the current monthly partition.
    Expects a CSV at data/<partition_key>.csv with standard Spotify columns.
    """
    partition_key = context.partition_key  # e.g. "2024-01-01"
    month = partition_key[:7]              # "2024-01"

    csv_path = os.path.join(DATA_DIR, f"{month}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No data file found for partition {month} at {csv_path}")

    df = pd.read_csv(csv_path)
    context.log.info(f"Loaded {len(df)} tracks for partition {month}")

    return Output(
        value=df,
        metadata={"num_tracks": len(df), "partition": month},
    )


# ---------------------------------------------------------------------------
# Asset 2: featured_tracks
# ---------------------------------------------------------------------------

@asset(partitions_def=monthly_partitions)
def featured_tracks(
    context: AssetExecutionContext,
    raw_tracks: pd.DataFrame,
    lakefs: LakeFSResource,
) -> Output[pd.DataFrame]:
    """
    Apply feature engineering to raw tracks and commit to LakeFS.

    BUG 3 (Logic — Wrong partition window, Medium):
    The partition_key from Dagster's MonthlyPartitionsDefinition is the *start*
    of the month (e.g. "2024-01-01"). The code below slices df using
    partition_key directly as a date string, which compares "track_date" against
    "2024-01-01" with a strict less-than, so it drops all tracks *on* the 1st of
    the month. The correct filter should be:
        month_start = pd.Timestamp(partition_key)
        month_end   = month_start + pd.offsets.MonthEnd(1)
        df = df[(df["track_date"] >= month_start) & (df["track_date"] <= month_end)]
    """
    partition_key = context.partition_key
    df = raw_tracks.copy()
    df["track_date"] = pd.to_datetime(df["track_date"])

    # Incorrect: uses strict less-than on the start date string, losing day-1 data
    month_start = pd.Timestamp(partition_key)
    month_end = month_start + pd.offsets.MonthEnd(1)
    df = df[(df["track_date"] > month_start) & (df["track_date"] <= month_end)]

    X, scaler = build_feature_matrix(df, fit_scaler=True)
    X["popularity"] = df["popularity"].values

    context.log.info(f"Engineered features for {len(X)} tracks")

    # Commit to LakeFS (will use target_branch from resource config)
    lakefs.commit_features(partition_key, message=f"Features for {partition_key}")

    return Output(
        value=X,
        metadata={"num_rows": len(X), "num_features": X.shape[1] - 1},
    )


# ---------------------------------------------------------------------------
# Asset 3: trained_model
# ---------------------------------------------------------------------------

@asset(partitions_def=monthly_partitions)
def trained_model(
    context: AssetExecutionContext,
    featured_tracks: pd.DataFrame,
    model_registry: ModelRegistryResource,
) -> Output[str]:
    """
    Train a RandomForestRegressor on the featured tracks partition.
    Saves the model to the model registry and returns the model path.
    """
    partition_key = context.partition_key

    X = featured_tracks.drop(columns=["popularity"])
    y = featured_tracks["popularity"]

    X_train, X_test, y_train, y_test = split_train_test(
        pd.concat([X, y], axis=1), test_size=0.2
    )
    y_train = X_train.pop("popularity") if "popularity" in X_train.columns else y_train
    y_test = X_test.pop("popularity") if "popularity" in X_test.columns else y_test

    model = train_model(X_train, y_train)

    os.makedirs(model_registry.registry_path, exist_ok=True)
    model_path = model_registry.model_path(partition_key)
    save_model(model, model_path)

    context.log.info(f"Model saved to {model_path}")

    return Output(
        value=model_path,
        metadata={"model_path": model_path, "partition": partition_key},
    )


# ---------------------------------------------------------------------------
# Asset 4: model_metrics
# ---------------------------------------------------------------------------

@asset(partitions_def=monthly_partitions)
def model_metrics(
    context: AssetExecutionContext,
    trained_model: str,
    featured_tracks: pd.DataFrame,
    model_registry: ModelRegistryResource,
) -> Output[dict]:
    """
    Evaluate the trained model and persist metrics as JSON.
    """
    from ml.model import load_model

    partition_key = context.partition_key
    model = load_model(trained_model)

    X = featured_tracks.drop(columns=["popularity"])
    y = featured_tracks["popularity"]

    _, X_test, _, y_test = split_train_test(
        pd.concat([X, y], axis=1), test_size=0.2
    )
    y_test = X_test.pop("popularity") if "popularity" in X_test.columns else y_test

    metrics = evaluate_model(model, X_test, y_test)

    metrics_path = model_registry.metrics_path(partition_key)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    context.log.info(f"Metrics: {metrics}")

    return Output(
        value=metrics,
        metadata={
            "mae": MetadataValue.float(metrics["mae"]),
            "r2": MetadataValue.float(metrics["r2"]),
        },
    )
