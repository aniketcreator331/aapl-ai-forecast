# AAPL AI Forecast

A Python project for **Apple (AAPL) stock analysis and next-day price prediction** using machine learning.

## Features

| Area | Details |
|------|---------|
| **Data** | Downloads historical OHLCV data from Yahoo Finance via `yfinance` |
| **Technical Indicators** | SMA (20/50), EMA (12/26), RSI (14), MACD, Bollinger Bands, OBV |
| **ML Model** | Random Forest Regressor trained on price + indicator features |
| **Prediction** | Predicts the next trading-day closing price |
| **Charts** | Saves publication-quality PNG charts to `charts/` |

## Project Structure

```
aapl-ai-forecast/
├── main.py                        # CLI entry point
├── requirements.txt
├── src/
│   ├── data_fetcher.py            # Download stock data (yfinance)
│   ├── technical_indicators.py    # Compute technical indicators (ta)
│   ├── model.py                   # Random Forest training & prediction
│   └── visualizer.py              # Matplotlib charts
└── tests/
    ├── test_technical_indicators.py
    └── test_model.py
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (default: AAPL, 2-year lookback)
python main.py

# 3. Optional arguments
python main.py --ticker MSFT --period 5y --output-dir output_charts
```

### Sample output

```
[1/5] Fetching AAPL data (2y) …
      504 trading days loaded  (2023-03-01 → 2025-02-28)
[2/5] Computing technical indicators …
[3/5] Training Random Forest model …
      Train rows : 384
      Test rows  : 96
      MAE        : $1.8734
      RMSE       : $2.4510
      R²         : 0.9821
[4/5] Predicting next-day closing price …
      Last close : $227.48
      Predicted  : $228.15  ▲ 0.29%
[5/5] Saving charts to 'charts/' …
  Chart saved → charts/moving_averages.png
  Chart saved → charts/bollinger_bands.png
  Chart saved → charts/rsi.png
  Chart saved → charts/macd.png
  Chart saved → charts/prediction.png

Done ✓
```

## Charts

| File | Description |
|------|-------------|
| `charts/moving_averages.png` | Close price with SMA-20/50 and EMA-12/26 |
| `charts/bollinger_bands.png` | Close price with Bollinger Band shading |
| `charts/rsi.png` | RSI (14) with overbought / oversold zones |
| `charts/macd.png` | MACD line, signal line and histogram |
| `charts/prediction.png` | Last 90 days + next-day price prediction |

## Running Tests

```bash
pytest tests/ -v
```

## Dependencies

- [yfinance](https://github.com/ranaroussi/yfinance) – market data
- [pandas](https://pandas.pydata.org/) – data manipulation
- [numpy](https://numpy.org/) – numerical computing
- [scikit-learn](https://scikit-learn.org/) – machine learning
- [matplotlib](https://matplotlib.org/) – charting
- [ta](https://technical-analysis-library-in-python.readthedocs.io/) – technical indicators
