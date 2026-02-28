import numpy as np

def predict_price(model, df):
    latest = df[["price", "MA_5", "MA_15", "momentum"]].iloc[-1:]
    prediction = model.predict(latest)
    return float(prediction[0])
