"""AAPL AI Forecast – main entry point.

Run the full pipeline:
  1. Download historical AAPL stock data.
  2. Compute technical indicators.
  3. Train a Random Forest model and evaluate it.
  4. Predict the next trading-day closing price.
  5. Save charts to the 'charts/' directory.

Usage
-----
    python main.py [--ticker AAPL] [--period 2y] [--output-dir charts]
"""

from __future__ import annotations

import argparse
import os
import sys

from src.data_fetcher import fetch_stock_data
from src.technical_indicators import add_all_indicators
from src.model import train_model, predict_next_close
from src.visualizer import (
    plot_price_and_moving_averages,
    plot_bollinger_bands,
    plot_rsi,
    plot_macd,
    plot_predictions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AAPL AI Forecast – Stock Analysis & Price Prediction")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--period", default="2y", help="Historical data period (default: 2y)")
    parser.add_argument("--output-dir", default="charts", help="Directory for saved charts (default: charts/)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ticker = args.ticker
    period = args.period
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Fetch data ────────────────────────────────────────────────────────
    print(f"[1/5] Fetching {ticker} data ({period}) …")
    try:
        df = fetch_stock_data(ticker=ticker, period=period)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"      {len(df)} trading days loaded  ({df.index[0].date()} → {df.index[-1].date()})")

    # ── 2. Technical indicators ───────────────────────────────────────────────
    print("[2/5] Computing technical indicators …")
    df = add_all_indicators(df)

    # ── 3. Train model ────────────────────────────────────────────────────────
    print("[3/5] Training Random Forest model …")
    model, scaler, metrics = train_model(df)
    print(f"      Train rows : {metrics['train_size']:,}")
    print(f"      Test rows  : {metrics['test_size']:,}")
    print(f"      MAE        : ${metrics['mae']:.4f}")
    print(f"      RMSE       : ${metrics['rmse']:.4f}")
    print(f"      R²         : {metrics['r2']:.4f}")

    # ── 4. Predict next close ─────────────────────────────────────────────────
    print("[4/5] Predicting next-day closing price …")
    predicted = predict_next_close(model, scaler, df)
    last_close = float(df["Close"].iloc[-1])
    change = predicted - last_close
    change_pct = (change / last_close) * 100
    direction = "▲" if change >= 0 else "▼"
    print(f"      Last close : ${last_close:.2f}")
    print(f"      Predicted  : ${predicted:.2f}  {direction} {abs(change_pct):.2f}%")

    # ── 5. Save charts ────────────────────────────────────────────────────────
    print(f"[5/5] Saving charts to '{output_dir}/' …")
    plot_price_and_moving_averages(df, ticker=ticker, save_path=os.path.join(output_dir, "moving_averages.png"))
    plot_bollinger_bands(df, ticker=ticker, save_path=os.path.join(output_dir, "bollinger_bands.png"))
    plot_rsi(df, ticker=ticker, save_path=os.path.join(output_dir, "rsi.png"))
    plot_macd(df, ticker=ticker, save_path=os.path.join(output_dir, "macd.png"))
    plot_predictions(df, predicted_close=predicted, ticker=ticker, save_path=os.path.join(output_dir, "prediction.png"))

    print("\nDone ✓")


if __name__ == "__main__":
    main()
