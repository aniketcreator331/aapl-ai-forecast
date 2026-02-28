"""
AAPL Price Prediction Model
Builds and trains an LSTM model to forecast next-day closing prices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


SEQUENCE_LEN = 60   # days of history used as input
FUTURE_DAYS = 30    # days to forecast into the future
TEST_RATIO = 0.15   # fraction of data held out for evaluation
EPOCHS = 50
BATCH_SIZE = 32
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def prepare_sequences(prices: np.ndarray, seq_len: int):
    """Create (X, y) sliding-window sequences from a 1-D price array."""
    X, y = [], []
    for i in range(seq_len, len(prices)):
        X.append(prices[i - seq_len:i, 0])
        y.append(prices[i, 0])
    return np.array(X), np.array(y)


def split_scale(close_series: pd.Series, seq_len: int = SEQUENCE_LEN, test_ratio: float = TEST_RATIO):
    """Scale data, create sequences, and split into train/test sets."""
    values = close_series.values.reshape(-1, 1)
    split_idx = int(len(values) * (1 - test_ratio))

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values[:split_idx])
    test_scaled = scaler.transform(values[split_idx - seq_len:])

    X_train, y_train = prepare_sequences(train_scaled, seq_len)
    X_test, y_test = prepare_sequences(test_scaled, seq_len)

    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, y_train, X_test, y_test, scaler, split_idx


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_model(seq_len: int = SEQUENCE_LEN) -> Sequential:
    """Build a two-layer stacked LSTM model."""
    model = Sequential([
        Input(shape=(seq_len, 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray) -> tf.keras.callbacks.History:
    """Train the model with early stopping."""
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray, scaler: MinMaxScaler):
    """Return actual vs predicted prices (inverse-scaled) and print metrics."""
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)

    print("\n=== Model Evaluation (Test Set) ===")
    print(f"MAE  : ${mae:.2f}")
    print(f"RMSE : ${rmse:.2f}")
    print(f"R²   : {r2:.4f}")
    print("=" * 36)

    return y_actual, y_pred


# ---------------------------------------------------------------------------
# Future forecast
# ---------------------------------------------------------------------------

def forecast_future(model: Sequential, last_sequence: np.ndarray, scaler: MinMaxScaler, days: int = FUTURE_DAYS) -> np.ndarray:
    """Autoregressively forecast `days` into the future."""
    seq = last_sequence.copy()   # shape: (seq_len,)
    preds = []
    for _ in range(days):
        x = seq.reshape(1, len(seq), 1)
        next_scaled = model.predict(x, verbose=0)[0, 0]
        preds.append(next_scaled)
        seq = np.append(seq[1:], next_scaled)

    future_prices = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return future_prices


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_predictions(
    close_series: pd.Series,
    split_idx: int,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    future_prices: np.ndarray,
    seq_len: int = SEQUENCE_LEN,
    save_path: str = "aapl_prediction.png",
) -> None:
    """Plot historical prices, test-set predictions, and future forecast."""
    test_start = split_idx
    test_dates = close_series.index[test_start:test_start + len(y_actual)]

    last_date = close_series.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(future_prices))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(close_series.index, close_series.values, label="Historical Close", color="steelblue", linewidth=1)
    ax.plot(test_dates, y_actual, label="Actual (Test)", color="green", linewidth=1.2)
    ax.plot(test_dates, y_pred, label="Predicted (Test)", color="orange", linewidth=1.2, linestyle="--")
    ax.plot(future_dates, future_prices, label=f"Forecast ({len(future_prices)}d)", color="red", linewidth=1.5, linestyle="--", marker="o", markersize=3)

    ax.axvline(x=close_series.index[test_start], color="grey", linestyle=":", linewidth=1, label="Train/Test Split")
    ax.set_title("AAPL Price Prediction (LSTM)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Model] Prediction chart saved to {save_path}")


def plot_training_loss(history: tf.keras.callbacks.History, save_path: str = "training_loss.png") -> None:
    """Plot training vs validation loss."""
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Model Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Model] Training loss chart saved to {save_path}")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_prediction(close_series: pd.Series):
    """End-to-end: train LSTM, evaluate, forecast, and plot."""
    print("[Model] Preparing data …")
    X_train, y_train, X_test, y_test, scaler, split_idx = split_scale(close_series)

    print(f"[Model] Training on {len(X_train)} samples, testing on {len(X_test)} samples …")
    model = build_model(seq_len=SEQUENCE_LEN)

    history = train_model(model, X_train, y_train)
    plot_training_loss(history)

    y_actual, y_pred = evaluate_model(model, X_test, y_test, scaler)

    # Use the last SEQUENCE_LEN scaled prices as seed for future forecast
    all_scaled = scaler.transform(close_series.values.reshape(-1, 1))
    last_seq = all_scaled[-SEQUENCE_LEN:, 0]
    future_prices = forecast_future(model, last_seq, scaler, days=FUTURE_DAYS)

    last_date = close_series.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=FUTURE_DAYS)
    print(f"\n[Model] {FUTURE_DAYS}-day forecast starting {future_dates[0].date()}:")
    for date, price in zip(future_dates, future_prices):
        print(f"  {date.date()} → ${price:.2f}")

    plot_predictions(close_series, split_idx, y_actual, y_pred, future_prices)

    return model, scaler, future_prices


if __name__ == "__main__":
    from stock_analysis import fetch_data, add_technical_indicators
    data = fetch_data()
    data = add_technical_indicators(data)
    run_prediction(data["Close"])
