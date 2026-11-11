# container: scipy
# Feature selection tip: add a few fake (random) features, fit a classifier
# on the data including them, and rank the features by importance. A real
# feature that ranks below the fakes is probably safe to drop.
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Example data: 8 features, but only the first 4 carry information about y
X, y = make_classification(n_samples=500, n_features=8, n_informative=4, n_redundant=0,
                           shuffle=False, random_state=0)
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])

# region: fit
# Fake features from various distributions
rng = np.random.default_rng(0)
X["fake_uniform"] = rng.uniform(size=len(X))
X["fake_normal"] = rng.normal(size=len(X))
X["fake_exp"] = rng.exponential(size=len(X))

# Fit a classifier INCLUDING the fakes
clf = RandomForestClassifier(random_state=0)
clf.fit(X, y)

# Rank the features, most important first
scores = clf.feature_importances_
rank = pd.Series(scores, index=X.columns)
rank = rank.sort_values(ascending=False)

# Real features below the best fake one
fake = rank.index.str.startswith("fake_")
best = rank[fake].max()
to_drop = rank[~fake & (rank < best)]
# endregion

print("Feature ranking:")
print(rank.round(3).to_string())
print("\nProbably safe to drop:", list(to_drop.index))
