"""
AAPL AI Forecast – Main Entry Point
Runs stock analysis and LSTM price prediction end-to-end.
"""

import sys
from stock_analysis import fetch_data, add_technical_indicators, plot_price_and_indicators, print_summary
from model import run_prediction


def main():
    print("=" * 50)
    print("  AAPL Stock Analysis & AI Price Prediction")
    print("=" * 50)

    # 1. Fetch and analyse data
    print("\n[Step 1/3] Downloading historical data …")
    df = fetch_data()
    df = add_technical_indicators(df)
    print_summary(df)

    # 2. Plot technical analysis chart
    print("\n[Step 2/3] Generating technical analysis chart …")
    plot_price_and_indicators(df, save_path="aapl_analysis.png")

    # 3. Train LSTM model and forecast
    print("\n[Step 3/3] Training LSTM model and forecasting prices …")
    close = df["Close"]
    run_prediction(close)

    print("\n✅  All done. Output charts:")
    print("   • aapl_analysis.png   – technical indicators")
    print("   • training_loss.png   – model training history")
    print("   • aapl_prediction.png – actual vs predicted + 30-day forecast")


if __name__ == "__main__":
    sys.exit(main())
