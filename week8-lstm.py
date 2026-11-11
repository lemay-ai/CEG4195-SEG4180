# container: tensorflow
# A small LSTM (Keras 3 on TensorFlow) forecasts the next 24 hours of temperature at
# Ottawa airport from the last 48 hours of weather, and so does the same LSTM with a skip
# connection (residual: it predicts the change from the last temperature). Both must beat
# two naive forecasts: "the last value stays" and "tomorrow repeats today".
# Data: hourly weather for 2024 and 2025 from Environment and Climate Change Canada
# (climate.weather.gc.ca, station OTTAWA INTL A, bulk CSV download on 2026-11-11), saved
# in data/week8-ottawa-weather-hourly.csv; the few missing hours are interpolated.
# Contains information licensed under the Open Government Licence - Canada.
# Full run (up to 30 epochs, early stopping), mean absolute error over 2025: last value
# 4.14 °C, tomorrow repeats today 3.72 °C, LSTM 2.71 °C, residual LSTM 2.52 °C.
# Tested November 2026: TensorFlow 2.21, Keras 3.15. COURSE_SMOKE=1 trains for 2 epochs.
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras import layers

WEATHER = Path(__file__).parent / "data" / "week8-ottawa-weather-hourly.csv"
EPOCHS = 2 if os.environ.get("COURSE_SMOKE") else 30
keras.utils.set_random_seed(0)

# Hourly temperature, humidity and pressure, plus the hour of day as a point on a circle
weather = pd.read_csv(WEATHER, index_col="time", parse_dates=True)
hour = weather.index.hour.to_numpy()
weather["hour_sin"] = np.sin(2 * np.pi * hour / 24)
weather["hour_cos"] = np.cos(2 * np.pi * hour / 24)

# Train on 2024, test on 2025. Scale with the training year's statistics only.
train, test = weather.loc["2024"], weather.loc["2025"]
mean, std = train.mean(), train.std()
train, test = (train - mean) / std, (test - mean) / std

past, future = 48, 24  # look at 48 hours, forecast the next 24


def windows(data):
    """Inputs: 48 hours of every feature. Targets: the next 24 hours of temperature."""
    values, temp = data.to_numpy(), data["temp_c"].to_numpy()
    starts = range(len(data) - past - future + 1)
    X = np.stack([values[i:i + past] for i in starts])
    y = np.stack([temp[i + past:i + past + future] for i in starts])
    return X, y  # X: (samples, 48 hours, 5 features), y: (samples, 24 hours)


X_train, y_train = windows(train)
X_test, y_test = windows(test)
n_features = X_train.shape[2]


# region: model
def lstm(residual=False):
    inputs = keras.Input(shape=(past, n_features))
    # Read the 48 hours, then predict the next 24
    memory = layers.LSTM(32)(inputs)
    outputs = layers.Dense(future)(memory)
    # Skip connection: add the last temperature seen,
    # so the LSTM only has to learn the change
    if residual:
        outputs = outputs + inputs[:, -1:, 0]
    return keras.Model(inputs, outputs)
# endregion


# region: train
models = {"LSTM": lstm(),
          "Residual LSTM": lstm(residual=True)}
for model in models.values():
    model.compile(optimizer="adam", loss="mse")
    # Validate on the end of 2024, in time order
    stop = keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=EPOCHS,
              batch_size=64, validation_split=0.2,
              callbacks=[stop], verbose=2)
# endregion


def degrees(scaled):
    """Back from scaled units to °C."""
    return scaled * std["temp_c"] + mean["temp_c"]


# Two naive forecasts, then the two LSTMs, scored on all of 2025
temp_now = X_test[:, -1, 0:1]  # the last hour seen
forecasts = {
    "Last value stays": np.repeat(temp_now, future, axis=1),
    "Tomorrow repeats today": X_test[:, -24:, 0],  # the last 24 hours again
}
for name, model in models.items():
    forecasts[name] = model.predict(X_test, verbose=0)
errors = {name: np.abs(degrees(f) - degrees(y_test)) for name, f in forecasts.items()}
for name, error in errors.items():
    print(f"{name:24s} mean absolute error: {error.mean():.2f} °C")

# Left: the error at each hour ahead, over all of 2025.
# Right: a typical day, the one where the LSTM's error is the median.
colours = {"Last value stays": "#2a78d6", "Tomorrow repeats today": "#eb6834",
           "LSTM": "#1baf7a", "Residual LSTM": "#eda100"}
styles = {"Last value stays": ":", "Tomorrow repeats today": "--", "LSTM": "-", "Residual LSTM": "-."}
hours_ahead = np.arange(1, future + 1)
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
for name, error in errors.items():
    left.plot(hours_ahead, error.mean(axis=0), styles[name], color=colours[name], lw=2, label=name)
left.set(xlabel="Hours ahead", ylabel="Mean absolute error (°C)", ylim=(0, None),
         title="Error by hour ahead, all of 2025")
left.legend(frameon=False)

lstm_error = errors["LSTM"].mean(axis=1)
i = np.argsort(lstm_error)[len(lstm_error) // 2]
now = test.index[i + past - 1]
right.plot(np.arange(-past + 1, 1), degrees(X_test[i, :, 0]), color="black", lw=1.5,
           label="Past 48 hours")
right.plot(hours_ahead, degrees(y_test[i]), color="grey", lw=1.5, label="What happened")
for name, forecast in forecasts.items():
    right.plot(hours_ahead, degrees(forecast[i]), styles[name], color=colours[name], lw=2)
right.axvline(0, color="grey", lw=0.5)
right.set(xlabel="Hours from now", ylabel="Temperature (°C)",
          title=f"A typical forecast, made {now:%Y-%m-%d %H:%M}")
right.legend(frameon=False)
for ax in (left, right):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e3", lw=0.8)
fig.tight_layout()
fig.savefig("week8-lstm.png", dpi=150)
plt.show()
