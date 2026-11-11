# container: scipy
# Active learning on scikit-learn's handwritten digits (8x8 images, a small MNIST).
# A logistic regression starts from 20 random labels. Each round it asks for 10 more,
# up to 400: either at random, or the 10 pool images it is least sure about
# (uncertainty sampling, least confident). The test set is 30% of the digits, and the
# score at each label count is the mean over 5 seeds. Labels come from the dataset,
# standing in for a person who labels the images the model asks for.
# Sample efficiency is the accuracy a method reaches for a given number of labels.
# See Settles, "Active Learning Literature Survey" (2009), section 3.1.
# Full run: random labels reach 0.917 at 200 and 0.947 at 400, uncertainty sampling 0.951
# at 200, and all 1,257 pool labels give 0.972 (about 10 seconds on a CPU).
# Tested November 2026: scikit-learn 1.9. COURSE_SMOKE=1 runs 1 seed up to 100 labels.
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SMOKE = os.environ.get("COURSE_SMOKE") == "1"
SEEDS = 1 if SMOKE else 5
START, STEP, BUDGET = 20, 10, 100 if SMOKE else 400

X, y = load_digits(return_X_y=True)
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
scaler = StandardScaler().fit(X_pool)
X_pool, X_test = scaler.transform(X_pool), scaler.transform(X_test)


def model():
    return LogisticRegression(max_iter=2000)


def least_confident(clf, X_unlabelled, k):
    """The k unlabelled images whose most likely class has the lowest probability."""
    confidence = clf.predict_proba(X_unlabelled).max(axis=1)
    return np.argsort(confidence)[:k]


def learning_curve(strategy, seed):
    rng = np.random.default_rng(seed)
    labelled = list(rng.choice(len(X_pool), START, replace=False))
    counts, scores = [], []
    while True:
        clf = model().fit(X_pool[labelled], y_pool[labelled])
        counts.append(len(labelled))
        scores.append(clf.score(X_test, y_test))
        if len(labelled) >= BUDGET:
            return np.array(counts), np.array(scores)
        unlabelled = np.setdiff1d(np.arange(len(X_pool)), labelled)
        if strategy == "uncertainty":
            picked = unlabelled[least_confident(clf, X_pool[unlabelled], STEP)]
        else:
            picked = rng.choice(unlabelled, STEP, replace=False)
        labelled += list(picked)


curves = {}
for strategy in ("uncertainty", "random"):
    runs = [learning_curve(strategy, seed) for seed in range(SEEDS)]
    counts = runs[0][0]
    curves[strategy] = (counts, np.mean([scores for _, scores in runs], axis=0))
full = model().fit(X_pool, y_pool).score(X_test, y_test)

print(f"Test accuracy, mean over {SEEDS} seed{'s' if SEEDS > 1 else ''} ({len(X_test)} test images)")
for n in (50, 100, 200, 300, 400):
    if n <= BUDGET:
        i = list(counts).index(n)
        print(f"{n:4d} labels: random {curves['random'][1][i]:.3f}, "
              f"uncertainty {curves['uncertainty'][1][i]:.3f}")
print(f"All {len(X_pool):,} pool labels: {full:.3f}")

fig, ax = plt.subplots(figsize=(6.4, 4.2))
names = {"uncertainty": "Uncertainty sampling", "random": "Random labels"}
colours = {"uncertainty": "#2a78d6", "random": "#eb6834"}
for strategy, (counts, scores) in curves.items():
    ax.plot(counts, scores, color=colours[strategy], lw=2, label=names[strategy])
# The point to see: uncertainty sampling with half the labels matches random at the budget
half = BUDGET // 2
i_half, i_end = list(counts).index(half), list(counts).index(BUDGET)
ax.plot([half, BUDGET], [curves["uncertainty"][1][i_half], curves["random"][1][i_end]], "o",
        color="black", ms=5, clip_on=False)
ax.annotate(f"{half} labels", (half, curves["uncertainty"][1][i_half]), xytext=(0, -15),
            textcoords="offset points", ha="center", fontsize=9)
ax.annotate(f"{BUDGET} labels", (BUDGET, curves["random"][1][i_end]), xytext=(-6, -14),
            textcoords="offset points", ha="right", fontsize=9)
ax.axhline(full, color="grey", ls="--", lw=1.2, label=f"All {len(X_pool):,} labels")
ax.set(xlabel="Labelled images", ylabel="Test accuracy", ylim=(0.5, 1.0), xlim=(0, BUDGET),
       title="Logistic regression on handwritten digits")
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e6e6e3", lw=0.8)
fig.tight_layout()
fig.savefig("week7-active-learning.png", dpi=150)
plt.show()
