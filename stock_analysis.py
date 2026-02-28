"""
AAPL Stock Analysis Module
Fetches historical data and computes technical indicators.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


TICKER = "AAPL"
START_DATE = "2015-01-01"


def fetch_data(ticker: str = TICKER, start: str = START_DATE) -> pd.DataFrame:
    """Download historical daily OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add commonly used technical indicators to the DataFrame."""
    close = df["Close"]

    # Moving averages
    df["MA_20"] = close.rolling(window=20).mean()
    df["MA_50"] = close.rolling(window=50).mean()
    df["MA_200"] = close.rolling(window=200).mean()

    # Bollinger Bands (20-day, 2 std)
    rolling_std = close.rolling(window=20).std()
    df["BB_Upper"] = df["MA_20"] + 2 * rolling_std
    df["BB_Lower"] = df["MA_20"] - 2 * rolling_std

    # RSI (14-day)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - 100 / (1 + rs)

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Daily returns
    df["Daily_Return"] = close.pct_change()

    return df


def plot_price_and_indicators(df: pd.DataFrame, save_path: str = "aapl_analysis.png") -> None:
    """Plot closing price with MAs, Bollinger Bands, RSI, and MACD."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("AAPL Stock Analysis", fontsize=16, fontweight="bold")

    # --- Price & MAs + Bollinger Bands ---
    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], label="Close", linewidth=1, color="steelblue")
    ax1.plot(df.index, df["MA_20"], label="MA 20", linewidth=1, linestyle="--", color="orange")
    ax1.plot(df.index, df["MA_50"], label="MA 50", linewidth=1, linestyle="--", color="green")
    ax1.plot(df.index, df["MA_200"], label="MA 200", linewidth=1, linestyle="--", color="red")
    ax1.fill_between(df.index, df["BB_Upper"], df["BB_Lower"], alpha=0.1, color="grey", label="Bollinger Bands")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- RSI ---
    ax2 = axes[1]
    ax2.plot(df.index, df["RSI"], label="RSI (14)", color="purple", linewidth=1)
    ax2.axhline(70, linestyle="--", color="red", linewidth=0.8, alpha=0.7)
    ax2.axhline(30, linestyle="--", color="green", linewidth=0.8, alpha=0.7)
    ax2.fill_between(df.index, df["RSI"], 70, where=(df["RSI"] >= 70), alpha=0.2, color="red")
    ax2.fill_between(df.index, df["RSI"], 30, where=(df["RSI"] <= 30), alpha=0.2, color="green")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- MACD ---
    ax3 = axes[2]
    ax3.plot(df.index, df["MACD"], label="MACD", color="blue", linewidth=1)
    ax3.plot(df.index, df["MACD_Signal"], label="Signal", color="orange", linewidth=1)
    ax3.bar(df.index, df["MACD_Hist"], label="Histogram", color=np.where(df["MACD_Hist"] >= 0, "green", "red"), alpha=0.5, width=1)
    ax3.set_ylabel("MACD")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Analysis] Chart saved to {save_path}")


def print_summary(df: pd.DataFrame) -> None:
    """Print a brief summary of the dataset."""
    close = df["Close"]
    print("\n=== AAPL Data Summary ===")
    print(f"Period      : {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} trading days)")
    print(f"Latest Close: ${float(close.iloc[-1]):.2f}")
    print(f"52-Week High: ${float(close[-252:].max()):.2f}")
    print(f"52-Week Low : ${float(close[-252:].min()):.2f}")
    ann_return = float((1 + df["Daily_Return"].mean()) ** 252 - 1) * 100
    ann_vol = float(df["Daily_Return"].std() * np.sqrt(252)) * 100
    print(f"Ann. Return : {ann_return:.1f}%")
    print(f"Ann. Volatility: {ann_vol:.1f}%")
    print("=" * 26)


if __name__ == "__main__":
    print("[Analysis] Fetching AAPL data …")
    data = fetch_data()
    data = add_technical_indicators(data)
    print_summary(data)
    plot_price_and_indicators(data)
