# container: scipy
"""Data leakage: the score a leaky evaluation reports, and the honest one.

Case 1: feature selection fitted on all the data before cross-validation.
The labels are random coin flips, so no model can beat 50% accuracy. Picking
the 20 features that best match the labels on all 200 rows lets the test
folds' labels into the training, and the score looks good. Inside a
Pipeline, the selection is refitted on each training fold only.

Case 2: the same person on both sides of the split. Each person gives 10
rows and has one label. KFold puts a person's rows in both training and
test, so the model recognizes the person. GroupKFold keeps each person on
one side.
"""
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# Case 1: 200 rows, 10,000 features of pure noise, random labels
X = rng.normal(size=(200, 10_000))
y = rng.integers(0, 2, size=200)

# region: leak
sel = SelectKBest(f_classif, k=20)
lr = LogisticRegression()

# Leaky: select using ALL rows first
X_20 = sel.fit_transform(X, y)
leaky = cross_val_score(lr, X_20, y, cv=cv)

# Honest: select inside each fold
model = make_pipeline(sel, lr)
honest = cross_val_score(model, X, y, cv=cv)

print(f"Leaky:  {leaky.mean():.0%}")
print(f"Honest: {honest.mean():.0%}")
# endregion

# Case 2: 40 people x 10 rows. Each person's rows sit close together and
# share that person's label, which is random.
people = np.repeat(np.arange(40), 10)
centres = rng.normal(size=(40, 5))
X2 = centres[people] + 0.1 * rng.normal(size=(400, 5))
y2 = rng.integers(0, 2, size=40)[people]

knn = KNeighborsClassifier(n_neighbors=1)
mixed = cross_val_score(knn, X2, y2, cv=cv)
grouped = cross_val_score(knn, X2, y2, cv=GroupKFold(n_splits=5),
                          groups=people)
print(f"KFold:      {mixed.mean():.0%}")
print(f"GroupKFold: {grouped.mean():.0%}")
