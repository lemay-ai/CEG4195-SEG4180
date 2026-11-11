# container: scipy
"""Fairness: precision and recall per group, not only over everyone.

The scikit-learn diabetes data (442 patients, bundled with scikit-learn, so
no download) gives a measure of disease progression after one year. The
label here is "progression above the median". The model never sees the sex
column, a common way to try to be fair. Every patient gets an
out-of-fold prediction from 5-fold cross-validation, then the metrics are
computed per sex group. The dataset codes sex as 1 or 2 and does not say
which is which.
"""
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option("display.precision", 2)
data = load_diabetes(as_frame=True, scaled=False).frame
sex = data["sex"].astype(int)
y = (data["target"] > data["target"].median()).astype(int)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression())

# region: groupby
def scores(g):  # the rows of one group
    y, p = g["y"], g["pred"]
    return pd.Series({
        "n": len(g), "base rate": y.mean(),
        "precision": precision_score(y, p),
        "recall": recall_score(y, p)})

X = data.drop(columns=["target", "sex"])
pred = cross_val_predict(model, X, y, cv=cv)
df = pd.DataFrame(dict(sex=sex, y=y, pred=pred))
table = df.groupby("sex").apply(scores)
print(table.astype({"n": int}))
# endregion
gap = table["recall"].max() - table["recall"].min()
print(f"Recall gap: {100 * gap:.0f} percentage points")

# The same model, given the sex column as a feature
pred = cross_val_predict(model, data.drop(columns=["target"]), y, cv=cv)
df = pd.DataFrame(dict(sex=sex, y=y, pred=pred))
table = df.groupby("sex").apply(scores)
gap = table["recall"].max() - table["recall"].min()
print(f"Recall gap with the sex column given to the model: {100 * gap:.0f} percentage points")
