import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

from data_collector import stream_data
from feature_engineering import create_features
from model import train_model
from predictor import predict_price

st.set_page_config(layout="wide")
st.title("📈 AAPL Real-Time AI Forecast (30s)")

st.write("Collecting live data...")

df = stream_data(90)  # collect 90 seconds
df = create_features(df)

if len(df) > 60:

    model = train_model(df)
    future_price = predict_price(model, df)

    current_price = df["price"].iloc[-1]

    # Metrics
    actual = df["price"].iloc[-30:]
    predicted = np.full(len(actual), future_price)

    mae = mean_absolute_error(actual, predicted)
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    # PRICE DISPLAY
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown(f"# ${current_price:.2f}")
    with col2:
        st.success("LIVE FEED ACTIVE")

    # CHART
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["price"],
        mode="lines",
        name="Live Price"
    ))

    fig.add_trace(go.Scatter(
        x=[df.index[-1], df.index[-1] + 30],
        y=[current_price, future_price],
        mode="lines",
        name="30s Forecast",
        line=dict(dash="dot")
    ))

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # METRICS
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")
    c3.metric("MAPE", f"{mape:.2f}%")

    # AI REASONING
    st.subheader("AI Intelligence")

    if df["MA_5"].iloc[-1] > df["MA_15"].iloc[-1]:
        trend = "Short-term MA crossed above long-term MA (Bullish signal)."
    else:
        trend = "Market consolidating."

    if df["momentum"].iloc[-1] > 0:
        momentum_text = "Positive short-term momentum detected."
    else:
        momentum_text = "Weak momentum."

    st.write("Trend Analysis:", trend)
    st.write("Momentum Analysis:", momentum_text)

else:
    st.warning("Insufficient data.")
