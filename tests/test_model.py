"""Tests for model module."""

import numpy as np
import pandas as pd
import pytest

from src.technical_indicators import add_all_indicators
from src.model import build_features, train_model, predict_next_close, FEATURE_COLS


def _make_df(n: int = 150) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with indicator columns."""
    rng = np.random.default_rng(0)
    close = 150 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(10_000_000, 100_000_000, n).astype(float)

    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )
    return add_all_indicators(df)


class TestBuildFeatures:
    def test_returns_tuple_of_df_and_series(self):
        X, y = build_features(_make_df())
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_no_nans_in_output(self):
        X, y = build_features(_make_df())
        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_X_y_same_length(self):
        X, y = build_features(_make_df())
        assert len(X) == len(y)

    def test_X_contains_expected_feature_cols(self):
        df = _make_df()
        available = [c for c in FEATURE_COLS if c in df.columns]
        X, _ = build_features(df)
        assert list(X.columns) == available


class TestTrainModel:
    def test_returns_three_values(self):
        result = train_model(_make_df())
        assert len(result) == 3

    def test_metrics_keys(self):
        _, _, metrics = train_model(_make_df())
        for key in ("mae", "rmse", "r2", "train_size", "test_size"):
            assert key in metrics

    def test_mae_positive(self):
        _, _, metrics = train_model(_make_df())
        assert metrics["mae"] >= 0

    def test_rmse_positive(self):
        _, _, metrics = train_model(_make_df())
        assert metrics["rmse"] >= 0

    def test_split_sizes_sum_to_total(self):
        df = _make_df()
        _, _, metrics = train_model(df, test_size=0.2)
        X, _ = build_features(df)
        assert metrics["train_size"] + metrics["test_size"] == len(X)


class TestPredictNextClose:
    def test_returns_float(self):
        df = _make_df()
        model, scaler, _ = train_model(df)
        result = predict_next_close(model, scaler, df)
        assert isinstance(result, float)

    def test_prediction_in_plausible_range(self):
        """Predicted price should be within 50 % of the last observed close."""
        df = _make_df()
        model, scaler, _ = train_model(df)
        predicted = predict_next_close(model, scaler, df)
        last_close = float(df["Close"].iloc[-1])
        assert last_close * 0.5 <= predicted <= last_close * 1.5
