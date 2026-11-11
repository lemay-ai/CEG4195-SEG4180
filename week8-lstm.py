# container: tensorflow
# needs: network
# A small LSTM (Keras 3 on TensorFlow): from ten one-minute closing prices of Alphabet
# (GOOG), predict whether the next minute closes higher, the task of week8-knn-dtw-minute.py.
# Prices come from Yahoo Finance (yfinance); without the network, from
# data/week8-goog-minute.csv (five days to 2026-11-10). Tested November 2026: yfinance 1.8.0,
# TensorFlow 2.21, Keras 3.15. COURSE_SMOKE=1 trains for 2 epochs instead of 10.
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf  # before keras: once TensorFlow is loaded, yfinance cannot download (ImpersonateError)
import keras
from keras import layers
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

SAVED_PRICES = Path(__file__).parent / "data" / "week8-goog-minute.csv"
EPOCHS = 2 if os.environ.get("COURSE_SMOKE") else 10


def closing_prices(ticker, **when):
    """Closing prices of one ticker: downloaded, or the saved copy when offline."""
    data = yf.download([ticker], progress=False, **when)
    if data.empty:  # no network
        print(f"Download failed: using the prices saved in {SAVED_PRICES.name}")
        return pd.read_csv(SAVED_PRICES, index_col=0)[ticker].dropna()
    return data["Close"][ticker].dropna()


# Download one-minute prices for the last five days
ticker = "GOOG"
prices = closing_prices(ticker, period="5d", interval="1m").to_numpy()

# Each sample is ten prices in a row; its label: is the next price higher than the last one?
window_size = 10
X, y = [], []
for j in range(len(prices) - window_size):
    X.append(prices[j:j + window_size])
    y.append(prices[j + window_size] > prices[j + window_size - 1])

X = np.array(X).reshape(-1, window_size, 1)  # (samples, time steps, features)
y = np.array(y)

# Split in time order: the first 80% to train
cutoff = int(len(X) * 0.8)
X_train, X_test = X[:cutoff], X[cutoff:]
y_train, y_test = y[:cutoff], y[cutoff:]

# Scale with the training set's statistics only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(X_test.shape)

# Define the LSTM model
model = keras.Sequential([
    keras.Input(shape=(window_size, 1)),
    layers.LSTM(10, activation="relu", return_sequences=True),
    layers.LSTM(5, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=2)

# Predict and evaluate
y_pred = model.predict(X_test, verbose=0).ravel() > 0.5
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, zero_division=0))
