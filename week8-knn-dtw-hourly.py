# container: scipy
# needs: network
# k-nearest neighbours with dynamic time warping (tslearn): from ten hourly closing prices
# of Alphabet (GOOG), predict whether the next hour closes higher. Prices come from Yahoo
# Finance (yfinance); without the network, from data/week8-goog-hourly.csv (three months
# to 2026-11-10). Tested November 2026: yfinance 1.8.0, tslearn 0.10.0, scikit-learn 1.9.1.
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

SAVED_PRICES = Path(__file__).parent / "data" / "week8-goog-hourly.csv"


def closing_prices(ticker, **when):
    """Closing prices of one ticker: downloaded, or the saved copy when offline."""
    data = yf.download([ticker], progress=False, **when)
    if data.empty:  # no network
        print(f"Download failed: using the prices saved in {SAVED_PRICES.name}")
        return pd.read_csv(SAVED_PRICES, index_col=0)[ticker].dropna()
    return data["Close"][ticker].dropna()


ticker = "GOOG"
period = "3mo"  # Also "1d", "5d", "1mo", ...
interval = "1h"  # Also "1m", "5m", "1d", ...
prices = closing_prices(ticker, period=period, interval=interval).to_numpy()

# Each sample is ten prices in a row.
# Its label: is the next price higher than the last one?
# region: windows
window_size = 10
X, y = [], []
for j in range(len(prices) - window_size):
    X.append(prices[j:j + window_size])
    y.append(prices[j + window_size] > prices[j + window_size - 1])

# tslearn wants (samples, time steps, features)
X = np.array(X).reshape(-1, window_size, 1)
y = np.array(y)
# endregion

# Split in time order: the first 80% to train
cutoff = int(len(X) * 0.8)
X_train, X_test = X[:cutoff], X[cutoff:]
y_train, y_test = y[:cutoff], y[cutoff:]

# Scale with the training set's statistics only
scaler = StandardScaler()
flat = X_train.reshape(len(X_train), -1)
X_train = scaler.fit_transform(flat).reshape(X_train.shape)
flat = X_test.reshape(len(X_test), -1)
X_test = scaler.transform(flat).reshape(X_test.shape)

# Train the kNN classifier with DTW as the distance
# region: knn
clf = KNeighborsTimeSeriesClassifier(metric="dtw")
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# endregion
print(classification_report(y_test, y_pred))
