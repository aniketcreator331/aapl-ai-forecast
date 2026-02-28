"""Train a Random Forest model to predict the next-day closing price of AAPL."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "sma_20", "sma_50", "ema_12", "ema_26",
    "rsi", "macd", "macd_signal", "macd_diff",
    "bb_upper", "bb_middle", "bb_lower", "obv",
]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target series y (next-day close).

    The target is the *next* day's closing price (shifted back by one row).
    Rows that have NaN in any feature or the target are dropped.

    Args:
        df: DataFrame that already contains indicator columns.

    Returns:
        Tuple of (X, y) with matching indices.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    data = df[available].copy()
    data["target"] = data["Close"].shift(-1)
    data.dropna(inplace=True)
    return data[available], data["target"]


def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, StandardScaler, dict]:
    """Train a Random Forest regressor to predict the next-day closing price.

    Args:
        df: DataFrame with price + indicator columns (output of add_all_indicators).
        test_size: Fraction of data reserved for evaluation (default 0.2).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (trained model, fitted scaler, metrics dict).
        Metrics dict contains: mae, rmse, r2, train_size, test_size.
    """
    X, y = build_features(df)

    # Chronological split – no shuffle to avoid look-ahead bias
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2_score(y_test, y_pred),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return model, scaler, metrics


def predict_next_close(
    model: RandomForestRegressor,
    scaler: StandardScaler,
    df: pd.DataFrame,
) -> float:
    """Predict the next trading day's closing price using the latest row of *df*.

    Args:
        model: Trained RandomForestRegressor.
        scaler: Fitted StandardScaler used during training.
        df: DataFrame with price + indicator columns (must contain the same
            feature columns used during training).

    Returns:
        Predicted closing price as a Python float.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    latest = df[available].dropna().iloc[[-1]]
    latest_scaled = scaler.transform(latest)
    return float(model.predict(latest_scaled)[0])
