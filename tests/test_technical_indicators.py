"""Tests for technical_indicators module."""

import numpy as np
import pandas as pd
import pytest

from src.technical_indicators import add_all_indicators


def _make_df(n: int = 100) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with *n* rows."""
    rng = np.random.default_rng(42)
    close = 150 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(10_000_000, 100_000_000, n).astype(float)

    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


class TestAddAllIndicators:
    def test_returns_dataframe(self):
        df = add_all_indicators(_make_df())
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        df = add_all_indicators(_make_df())
        expected = [
            "sma_20", "sma_50", "ema_12", "ema_26",
            "rsi",
            "macd", "macd_signal", "macd_diff",
            "bb_upper", "bb_middle", "bb_lower",
            "obv",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self):
        df = add_all_indicators(_make_df())
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns

    def test_sma_20_correct_window(self):
        """The first 19 SMA-20 values should be NaN; index 19 onward should not."""
        df = add_all_indicators(_make_df(60))
        assert df["sma_20"].iloc[:19].isna().all()
        assert not df["sma_20"].iloc[19:].isna().any()

    def test_sma_values_are_reasonable(self):
        """SMA-20 should be close to the mean of the surrounding prices."""
        df = add_all_indicators(_make_df())
        row = 50
        manual_sma = df["Close"].iloc[row - 19: row + 1].mean()
        assert abs(df["sma_20"].iloc[row] - manual_sma) < 1e-6

    def test_rsi_bounds(self):
        """RSI must lie in [0, 100]."""
        df = add_all_indicators(_make_df())
        rsi = df["rsi"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_bollinger_band_ordering(self):
        """Upper band >= middle band >= lower band (after warm-up period)."""
        df = add_all_indicators(_make_df())
        valid = df[["bb_upper", "bb_middle", "bb_lower"]].dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_modifies_in_place_and_returns_same_object(self):
        df = _make_df()
        result = add_all_indicators(df)
        assert result is df
