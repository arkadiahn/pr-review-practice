"""
Unit tests for feature engineering logic.
Run with: pytest tests/
"""

import pytest
import numpy as np
import pandas as pd

from ml.features import build_feature_matrix, split_train_test, ALL_FEATURES


def _make_sample_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Create a minimal synthetic track DataFrame for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "danceability":      rng.uniform(0, 1, n),
        "energy":            rng.uniform(0, 1, n),
        "loudness":          rng.uniform(-30, 0, n),
        "speechiness":       rng.uniform(0, 1, n),
        "acousticness":      rng.uniform(0, 1, n),
        "instrumentalness":  rng.uniform(0, 1, n),
        "liveness":          rng.uniform(0, 1, n),
        "valence":           rng.uniform(0, 1, n),
        "tempo":             rng.uniform(60, 200, n),
        "duration_ms":       rng.integers(120_000, 360_000, n),
        "popularity":        rng.integers(0, 100, n),
        "explicit":          rng.choice([True, False], n),
        "key":               rng.integers(0, 11, n),
        "mode":              rng.integers(0, 1, n),
        "time_signature":    rng.integers(3, 5, n),
    })


class TestBuildFeatureMatrix:

    def test_returns_dataframe_and_scaler(self):
        df = _make_sample_df()
        X, scaler = build_feature_matrix(df, fit_scaler=True)
        assert isinstance(X, pd.DataFrame)
        assert X.shape[0] == len(df)

    def test_scaled_mean_near_zero(self):
        """After StandardScaler, column means should be close to 0."""
        df = _make_sample_df(n=100)
        X, _ = build_feature_matrix(df, fit_scaler=True)
        assert np.allclose(X.mean(), 0, atol=1e-6)

    def test_derived_features_present(self):
        """Derived columns should be present in output."""
        df = _make_sample_df()
        X, _ = build_feature_matrix(df, fit_scaler=True)
        assert "energy_danceability_ratio" in X.columns
        assert "duration_minutes" in X.columns
        assert "explicit_int" in X.columns

    def test_no_missing_values(self):
        df = _make_sample_df()
        X, _ = build_feature_matrix(df, fit_scaler=True)
        assert not X.isnull().any().any()

    # BUG 6 (Logic — Wrong expected value in test, Easy):
    # duration_minutes = duration_ms / 60_000.
    # A track of 180_000 ms → 180_000 / 60_000 = 3.0 minutes.
    # The assertion below expects 2.0, which will always fail.
    # Should be: assert duration_minutes == pytest.approx(3.0)
    def test_duration_minutes_conversion(self):
        """duration_minutes should be duration_ms divided by 60_000."""
        df = _make_sample_df(n=1)
        df["duration_ms"] = 180_000  # exactly 3 minutes

        # We need unscaled values to check the raw conversion
        df_copy = df.copy()
        df_copy["duration_minutes"] = df_copy["duration_ms"] / 60_000

        duration_minutes = df_copy["duration_minutes"].iloc[0]
        assert duration_minutes == pytest.approx(2.0)   # BUG: should be 3.0


class TestSplitTrainTest:

    def test_split_sizes(self):
        df = _make_sample_df(n=100)
        train, test = split_train_test(df, test_size=0.2)
        assert len(test) == 20
        assert len(train) == 80

    def test_no_overlap(self):
        df = _make_sample_df(n=100)
        train, test = split_train_test(df, test_size=0.2)
        train_idx = set(train.index)
        test_idx = set(test.index)
        assert train_idx.isdisjoint(test_idx)

    def test_deterministic_with_seed(self):
        df = _make_sample_df(n=50)
        train1, test1 = split_train_test(df, random_state=42)
        train2, test2 = split_train_test(df, random_state=42)
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)
