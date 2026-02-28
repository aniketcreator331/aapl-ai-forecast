"""Visualise AAPL price data, technical indicators and model predictions."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe for both CLI and CI

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_price_and_moving_averages(df: pd.DataFrame, ticker: str = "AAPL", save_path: str | None = None) -> None:
    """Plot closing price with SMA-20, SMA-50, EMA-12 and EMA-26 overlaid.

    Args:
        df: DataFrame with Close, sma_20, sma_50, ema_12, ema_26 columns.
        ticker: Stock ticker label for the chart title.
        save_path: If provided, save the figure to this file path instead of
                   displaying it interactively.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["Close"], label="Close", linewidth=1.5, color="black")

    for col, label, color in [
        ("sma_20", "SMA 20", "blue"),
        ("sma_50", "SMA 50", "orange"),
        ("ema_12", "EMA 12", "green"),
        ("ema_26", "EMA 26", "red"),
    ]:
        if col in df.columns:
            ax.plot(df.index, df[col], label=label, linewidth=1, linestyle="--", color=color)

    ax.set_title(f"{ticker} – Closing Price & Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_or_show(fig, save_path)


def plot_bollinger_bands(df: pd.DataFrame, ticker: str = "AAPL", save_path: str | None = None) -> None:
    """Plot closing price with Bollinger Bands shading.

    Args:
        df: DataFrame with Close, bb_upper, bb_middle, bb_lower columns.
        ticker: Stock ticker label.
        save_path: Optional output file path.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["Close"], label="Close", linewidth=1.5, color="black")

    if {"bb_upper", "bb_middle", "bb_lower"}.issubset(df.columns):
        ax.plot(df.index, df["bb_upper"], label="Upper Band", linewidth=1, color="blue")
        ax.plot(df.index, df["bb_middle"], label="Middle Band (SMA 20)", linewidth=1, linestyle="--", color="grey")
        ax.plot(df.index, df["bb_lower"], label="Lower Band", linewidth=1, color="blue")
        ax.fill_between(df.index, df["bb_lower"], df["bb_upper"], alpha=0.1, color="blue")

    ax.set_title(f"{ticker} – Bollinger Bands")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_or_show(fig, save_path)


def plot_rsi(df: pd.DataFrame, ticker: str = "AAPL", save_path: str | None = None) -> None:
    """Plot the Relative Strength Index (RSI) with overbought/oversold zones.

    Args:
        df: DataFrame with rsi column.
        ticker: Stock ticker label.
        save_path: Optional output file path.
    """
    if "rsi" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["rsi"], label="RSI (14)", linewidth=1.2, color="purple")
    ax.axhline(70, linestyle="--", color="red", linewidth=1, label="Overbought (70)")
    ax.axhline(30, linestyle="--", color="green", linewidth=1, label="Oversold (30)")
    ax.fill_between(df.index, 30, df["rsi"], where=(df["rsi"] < 30), alpha=0.3, color="green")
    ax.fill_between(df.index, 70, df["rsi"], where=(df["rsi"] > 70), alpha=0.3, color="red")

    ax.set_title(f"{ticker} – RSI (14)")
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_or_show(fig, save_path)


def plot_macd(df: pd.DataFrame, ticker: str = "AAPL", save_path: str | None = None) -> None:
    """Plot the MACD line, signal line and histogram.

    Args:
        df: DataFrame with macd, macd_signal and macd_diff columns.
        ticker: Stock ticker label.
        save_path: Optional output file path.
    """
    if not {"macd", "macd_signal", "macd_diff"}.issubset(df.columns):
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["macd"], label="MACD", linewidth=1.2, color="blue")
    ax.plot(df.index, df["macd_signal"], label="Signal", linewidth=1.2, color="orange")
    colors = ["green" if v >= 0 else "red" for v in df["macd_diff"]]
    ax.bar(df.index, df["macd_diff"], color=colors, alpha=0.4, label="Histogram")
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title(f"{ticker} – MACD")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_or_show(fig, save_path)


def plot_predictions(
    df: pd.DataFrame,
    predicted_close: float,
    ticker: str = "AAPL",
    save_path: str | None = None,
) -> None:
    """Plot the last 90 days of closing price alongside the next-day prediction.

    Args:
        df: Full DataFrame with Close column.
        predicted_close: Predicted next-day closing price.
        ticker: Stock ticker label.
        save_path: Optional output file path.
    """
    recent = df["Close"].tail(90)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(recent.index, recent.values, label="Historical Close", linewidth=1.5, color="black")

    # Show the prediction one bar ahead of the last historical date
    last_date = recent.index[-1]
    ax.scatter([last_date], [predicted_close], color="red", zorder=5, s=80, label=f"Predicted: ${predicted_close:.2f}")
    ax.annotate(
        f"${predicted_close:.2f}",
        xy=(last_date, predicted_close),
        xytext=(10, 10),
        textcoords="offset points",
        color="red",
        fontsize=10,
    )

    ax.set_title(f"{ticker} – Next-Day Close Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    """Save figure to *save_path* or show it interactively."""
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Chart saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
