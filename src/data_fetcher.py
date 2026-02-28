"""Fetch historical AAPL stock data using yfinance."""

import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str = "AAPL", period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Download historical OHLCV data for the given ticker.

    Args:
        ticker: Stock ticker symbol (default: 'AAPL').
        period: Lookback period understood by yfinance, e.g. '1y', '2y', '5y'.
        interval: Bar interval, e.g. '1d', '1wk'.

    Returns:
        DataFrame with columns Open, High, Low, Close, Volume indexed by Date.

    Raises:
        ValueError: If no data is returned for the given ticker/period.
    """
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' with period='{period}'.")

    # Flatten multi-level columns produced by yfinance when downloading a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]]
