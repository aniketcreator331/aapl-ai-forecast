"""Compute common technical indicators on OHLCV price data."""

import pandas as pd
import ta


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a standard set of technical indicators to *df* in-place and return it.

    The following indicators are added as new columns:

    Trend
    -----
    sma_20, sma_50  – Simple Moving Averages (20-day and 50-day)
    ema_12, ema_26  – Exponential Moving Averages (12-day and 26-day)

    Momentum
    --------
    rsi             – Relative Strength Index (14-day)
    macd            – MACD line (EMA12 - EMA26)
    macd_signal     – Signal line (9-day EMA of MACD)
    macd_diff       – MACD histogram

    Volatility
    ----------
    bb_upper        – Bollinger Band upper band (20-day, 2 std)
    bb_middle       – Bollinger Band middle band
    bb_lower        – Bollinger Band lower band

    Volume
    ------
    obv             – On-Balance Volume

    Args:
        df: DataFrame containing at least Close, High, Low, Volume columns.

    Returns:
        The same DataFrame with new indicator columns appended.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Trend – moving averages
    df["sma_20"] = ta.trend.sma_indicator(close, window=20)
    df["sma_50"] = ta.trend.sma_indicator(close, window=50)
    df["ema_12"] = ta.trend.ema_indicator(close, window=12)
    df["ema_26"] = ta.trend.ema_indicator(close, window=26)

    # Momentum
    df["rsi"] = ta.momentum.rsi(close, window=14)

    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"] = macd_obj.macd_diff()

    # Volatility – Bollinger Bands
    bb_obj = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb_obj.bollinger_hband()
    df["bb_middle"] = bb_obj.bollinger_mavg()
    df["bb_lower"] = bb_obj.bollinger_lband()

    # Volume
    df["obv"] = ta.volume.on_balance_volume(close, volume)

    return df
