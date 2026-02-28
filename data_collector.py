import requests
import pandas as pd
import time

API_KEY = "YOUR_FINNHUB_API_KEY"
BASE_URL = "https://finnhub.io/api/v1/quote"

def get_live_price():
    url = f"{BASE_URL}?symbol=AAPL&token={API_KEY}"
    response = requests.get(url)
    data = response.json()

    return {
        "price": data["c"],
        "volume": data.get("v", 0),
        "timestamp": time.time()
    }

def stream_data(duration=60):
    records = []
    for _ in range(duration):
        tick = get_live_price()
        records.append(tick)
        time.sleep(1)

    df = pd.DataFrame(records)
    return df
