# container: scipy
# needs: network
# SARIMAX on Apple's daily closing price, 2020 to 2022, with the volatility index (VIX)
# as the exogenous variable, from Yahoo Finance (yfinance). Without the network it reads
# the same prices, saved on 2026-11-11, from data/week8-aapl-vix-daily.csv.
# Tested November 2026: yfinance 1.8.0, statsmodels 0.15.0.
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX

SAVED_PRICES = Path(__file__).parent / "data" / "week8-aapl-vix-daily.csv"


def closing_prices(tickers, **dates):
    """Daily closing prices, one column per ticker: downloaded, or the saved copy when offline."""
    data = yf.download(tickers, progress=False, **dates)
    if data.empty:  # no network
        print(f"Download failed: using the prices saved in {SAVED_PRICES.name}")
        return pd.read_csv(SAVED_PRICES, index_col=0, parse_dates=True)[tickers]
    return data["Close"][tickers]


# Closing prices of Apple and the VIX, on the dates both have (columns "AAPL" and "^VIX")
df = closing_prices(["AAPL", "^VIX"], start="2020-01-01", end="2023-01-01").dropna()

# Fit the SARIMA model with VIX as an exogenous variable.
# The data are daily trading prices, so the season s = 5 is one trading week.
model = SARIMAX(df["AAPL"], exog=df["^VIX"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
model_fit = model.fit(disp=False)

# One-step-ahead predictions for the last 30 trading days
predictions = model_fit.predict(start=len(df) - 30, end=len(df) - 1)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df["AAPL"].tail(30), label="Actual AAPL")
plt.plot(df["^VIX"].tail(30), label="Actual ^VIX")
plt.plot(predictions, label="Predicted AAPL", color="red")
plt.legend()
plt.title("SARIMAX Model Predictions for AAPL with ^VIX")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.show()

print(model_fit.summary())
