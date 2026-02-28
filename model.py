from sklearn.linear_model import LinearRegression
import numpy as np

def train_model(df):
    X = df[["price", "MA_5", "MA_15", "momentum"]]
    y = df["price"].shift(-30)

    X = X[:-30]
    y = y[:-30]

    model = LinearRegression()
    model.fit(X, y)

    return model
