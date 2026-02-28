# aapl-ai-forecast

**AAPL Stock Analysis & AI Price Prediction**

A Python project that fetches Apple Inc. (AAPL) historical stock data, computes key technical indicators, and trains an LSTM neural-network model to forecast future closing prices.

---

## Features

| Area | Details |
|---|---|
| **Data** | Daily OHLCV data from Yahoo Finance via `yfinance` |
| **Technical Indicators** | 20/50/200-day Moving Averages, Bollinger Bands, RSI (14), MACD |
| **Model** | Two-layer stacked LSTM with dropout regularisation |
| **Forecast** | 30-day autoregressive price forecast |
| **Output Charts** | `aapl_analysis.png`, `training_loss.png`, `aapl_prediction.png` |

---

## Project Structure

```
aapl-ai-forecast/
├── app.py               # Main entry point
├── stock_analysis.py    # Data fetching & technical indicators
├── model.py             # LSTM model, training & forecasting
├── requirements.txt     # Python dependencies
└── tests/
    └── test_stock.py    # Unit tests (no network required)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (downloads data, trains model, saves charts)
python app.py
```

### Run only analysis
```bash
python stock_analysis.py
```

### Run only model training & forecasting
```bash
python model.py
```

### Run tests
```bash
pytest tests/
```

---

## Output

After running `app.py` three PNG files are saved to the working directory:

* **`aapl_analysis.png`** – Closing price with Moving Averages, Bollinger Bands, RSI and MACD panels.
* **`training_loss.png`** – Train vs validation MSE loss curve.
* **`aapl_prediction.png`** – Actual vs predicted prices on the test set, plus a 30-day future forecast.

---

## Model Architecture

```
Input  →  LSTM(128, return_sequences=True)  →  Dropout(0.2)
       →  LSTM(64)                           →  Dropout(0.2)
       →  Dense(32, relu)                   →  Dense(1)
```

* **Loss**: Mean Squared Error  
* **Optimizer**: Adam  
* **Input window**: 60 trading days  
* **Early stopping**: patience = 10 epochs (val_loss)

---

## Dependencies

See `requirements.txt` for pinned versions.  Core libraries:

* `yfinance` – market data
* `pandas`, `numpy` – data manipulation
* `matplotlib`, `seaborn` – visualisation
* `scikit-learn` – scaling & metrics
* `tensorflow` / `keras` – LSTM model
