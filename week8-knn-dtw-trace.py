# container: scipy
# Event classification with k-nearest neighbours and dynamic time warping (tslearn).
# Trace is a benchmark of simulated sensor transients from a nuclear power plant
# (Roverso 2000, from the UCR archive): 4 kinds of event, 275 readings each, and the
# event starts at a different time in each series. tslearn ships a copy, so no network.
# Euclidean distance compares reading i with reading i; DTW lets the series stretch in
# time before it compares them, so a late event still matches an early one.
# Tested November 2026: tslearn 0.10.0, scikit-learn 1.9.1, matplotlib 3.11.2.
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tslearn.datasets import CachedDatasets
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

# X_train and X_test: (100, 275, 1), that is series, readings, features
# region: knn
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
for metric in ["euclidean", "dtw"]:
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=metric)
    clf.fit(X_train, y_train)
    print(metric, accuracy_score(y_test, clf.predict(X_test)))
# endregion

# Full DTW compares every reading with every other. A Sakoe-Chiba window of 10 readings
# only lets the match shift by 10: nearly as accurate, and faster.
for name, params in [("full", None), ("window 10", {"global_constraint": "sakoe_chiba",
                                                     "sakoe_chiba_radius": 10})]:
    start = time.perf_counter()
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw", metric_params=params)
    accuracy = accuracy_score(y_test, clf.fit(X_train, y_train).predict(X_test))
    print(f"DTW, {name:9s} accuracy {accuracy:.2f}  ({time.perf_counter() - start:.1f} s)")

# Which events does Euclidean distance get wrong?
clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="euclidean").fit(X_train, y_train)
print(confusion_matrix(y_test, clf.predict(X_test)))

# Three events of each class: the same shape, starting at different times
fig, axes = plt.subplots(1, 4, figsize=(12, 2.4), sharey=True)
for label, ax in zip(np.unique(y_train), axes):
    for series in X_train[y_train == label][:3]:
        ax.plot(series[:, 0], lw=1.2)
    ax.set_title(f"Class {label}")
    ax.set_xlabel("Reading")
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("week8-knn-dtw-trace.png", dpi=150)
plt.show()
