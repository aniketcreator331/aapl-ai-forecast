"""
Unit tests for stock_analysis.py and model.py.
Runs without network access – uses synthetic data where needed.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_series(n: int = 300, seed: int = 0) -> pd.Series:
    """Return a synthetic Close price series with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    prices = 150.0 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.bdate_range(start="2020-01-02", periods=n)
    return pd.Series(prices, index=dates, name="Close")


def _make_ohlcv_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    close = _make_price_series(n, seed)
    df = pd.DataFrame({
        "Open": close * 0.99,
        "High": close * 1.01,
        "Low": close * 0.98,
        "Close": close,
        "Volume": np.random.randint(50_000_000, 100_000_000, n),
    }, index=close.index)
    return df


# ---------------------------------------------------------------------------
# stock_analysis tests
# ---------------------------------------------------------------------------

class TestTechnicalIndicators:
    def test_indicator_columns_added(self):
        from stock_analysis import add_technical_indicators
        df = _make_ohlcv_df()
        result = add_technical_indicators(df)
        expected = ["MA_20", "MA_50", "MA_200", "BB_Upper", "BB_Lower", "RSI", "MACD", "MACD_Signal", "MACD_Hist", "Daily_Return"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_ma20_length(self):
        from stock_analysis import add_technical_indicators
        df = _make_ohlcv_df()
        result = add_technical_indicators(df)
        # First 19 values should be NaN, rest should be finite
        assert result["MA_20"].iloc[:19].isna().all()
        assert result["MA_20"].iloc[19:].notna().all()

    def test_rsi_bounds(self):
        from stock_analysis import add_technical_indicators
        df = _make_ohlcv_df()
        result = add_technical_indicators(df)
        rsi = result["RSI"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI must be in [0, 100]"

    def test_bollinger_bands_relationship(self):
        from stock_analysis import add_technical_indicators
        df = _make_ohlcv_df()
        result = add_technical_indicators(df).dropna()
        assert (result["BB_Upper"] >= result["MA_20"]).all()
        assert (result["BB_Lower"] <= result["MA_20"]).all()

    def test_daily_return_shape(self):
        from stock_analysis import add_technical_indicators
        df = _make_ohlcv_df(n=100)
        result = add_technical_indicators(df)
        assert len(result["Daily_Return"]) == 100


# ---------------------------------------------------------------------------
# model.py tests
# ---------------------------------------------------------------------------

class TestDataPreparation:
    def test_prepare_sequences_shape(self):
        from model import prepare_sequences
        prices = np.linspace(0, 1, 100).reshape(-1, 1)
        X, y = prepare_sequences(prices, seq_len=10)
        assert X.shape == (90, 10)
        assert y.shape == (90,)

    def test_prepare_sequences_values(self):
        from model import prepare_sequences
        prices = np.arange(20).reshape(-1, 1).astype(float)
        X, y = prepare_sequences(prices, seq_len=5)
        # First window should be [0,1,2,3,4] and target 5
        np.testing.assert_array_equal(X[0], [0, 1, 2, 3, 4])
        assert y[0] == 5

    def test_split_scale_shapes(self):
        from model import split_scale
        close = _make_price_series(n=300)
        seq_len = 60
        X_train, y_train, X_test, y_test, scaler, split_idx = split_scale(close, seq_len=seq_len, test_ratio=0.15)
        # LSTM input must be 3-D
        assert X_train.ndim == 3 and X_train.shape[2] == 1
        assert X_test.ndim == 3 and X_test.shape[2] == 1
        # Targets must be 1-D
        assert y_train.ndim == 1
        assert y_test.ndim == 1

    def test_split_scale_scaled_range(self):
        from model import split_scale
        close = _make_price_series(n=300)
        X_train, y_train, *_ = split_scale(close)
        # Training data is MinMax-scaled to [0, 1]
        assert X_train.min() >= 0.0 - 1e-9
        assert X_train.max() <= 1.0 + 1e-9

    def test_split_ratio(self):
        from model import split_scale
        close = _make_price_series(n=400)
        _, y_train, _, y_test, _, split_idx = split_scale(close, test_ratio=0.20)
        assert split_idx == int(400 * 0.80)


class TestBuildModel:
    def test_model_output_shape(self):
        from model import build_model, SEQUENCE_LEN
        model = build_model(seq_len=SEQUENCE_LEN)
        dummy = np.zeros((4, SEQUENCE_LEN, 1))
        out = model.predict(dummy, verbose=0)
        assert out.shape == (4, 1)

    def test_model_is_compiled(self):
        from model import build_model
        model = build_model()
        # A compiled Keras model has an optimizer attribute set
        assert model.optimizer is not None


class TestForecast:
    def test_forecast_length(self):
        from model import build_model, forecast_future, SEQUENCE_LEN
        from sklearn.preprocessing import MinMaxScaler
        model = build_model(seq_len=SEQUENCE_LEN)
        scaler = MinMaxScaler()
        dummy_prices = np.linspace(100, 200, 500).reshape(-1, 1)
        scaler.fit(dummy_prices)
        last_seq = scaler.transform(dummy_prices[-SEQUENCE_LEN:]).flatten()
        future = forecast_future(model, last_seq, scaler, days=10)
        assert len(future) == 10

    def test_forecast_positive_prices(self):
        from model import build_model, forecast_future, SEQUENCE_LEN
        from sklearn.preprocessing import MinMaxScaler
        model = build_model(seq_len=SEQUENCE_LEN)
        scaler = MinMaxScaler()
        dummy_prices = np.linspace(100, 200, 500).reshape(-1, 1)
        scaler.fit(dummy_prices)
        last_seq = scaler.transform(dummy_prices[-SEQUENCE_LEN:]).flatten()
        future = forecast_future(model, last_seq, scaler, days=5)
        assert (future > 0).all(), "Forecasted prices should be positive"
