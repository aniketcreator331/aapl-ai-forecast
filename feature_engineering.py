import pandas as pd

def create_features(df):
    df["MA_5"] = df["price"].rolling(5).mean()
    df["MA_15"] = df["price"].rolling(15).mean()
    df["momentum"] = df["price"].diff()
    df.dropna(inplace=True)
    return df
