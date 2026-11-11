# container: scipy
# needs: network
# ARIMA on Apple's daily closing price, 2020 to 2022, from Yahoo Finance (yfinance).
# Without the network it reads the same prices, saved on 2026-11-11, from
# data/week8-aapl-vix-daily.csv. Tested November 2026: yfinance 1.8.0, statsmodels 0.15.0.
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

SAVED_PRICES = Path(__file__).parent / "data" / "week8-aapl-vix-daily.csv"


def closing_prices(tickers, **dates):
    """Daily closing prices, one column per ticker: downloaded, or the saved copy when offline."""
    data = yf.download(tickers, progress=False, **dates)
    if data.empty:  # no network
        print(f"Download failed: using the prices saved in {SAVED_PRICES.name}")
        return pd.read_csv(SAVED_PRICES, index_col=0, parse_dates=True)[tickers]
    return data["Close"][tickers]


# Apple's closing price, 2020 to 2022
close = closing_prices(["AAPL"], start="2020-01-01", end="2023-01-01")["AAPL"]

# Fit the ARIMA model
model = ARIMA(close, order=(5, 1, 0))  # Example order, adjust as needed
model_fit = model.fit()

# One-step-ahead predictions for the last 30 trading days
predictions = model_fit.predict(start=len(close) - 30, end=len(close) - 1)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(close.tail(30), label="Actual")
plt.plot(predictions, label="Predicted", color="red")
plt.legend()
plt.title("ARIMA Model Predictions for AAPL")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.show()

print(model_fit.summary())
