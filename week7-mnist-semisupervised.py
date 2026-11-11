# container: scipy
# Semisupervised learning on scikit-learn's handwritten digits (8x8 images, a small
# MNIST): train on 10% of the labels, then let LabelSpreading use the other 80% unlabelled.
import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from icecream import ic
from sklearn.ensemble import RandomForestClassifier

# Load the digits dataset (1,797 images)
digits = datasets.load_digits()

X = digits.data
y = digits.target
images = digits.images

# Split of unlabelled 80%, train 10%, test 10%
X_un, X_rest, y_un, y_rest, images_un, images_rest = train_test_split(
    X, y, images, test_size=0.2, random_state=42, shuffle=True)
X_test, X_train, y_test, y_train, images_test, images_train = train_test_split(
    X_rest, y_rest, images_rest, test_size=0.5, random_state=42, shuffle=True)

ic(X_train.shape, y_train.shape, images_train.shape)
ic(X_test.shape, y_test.shape, images_test.shape)
ic(X_un.shape, y_un.shape, images_un.shape)

# Baseline: a random forest trained on the labelled 10% only
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_predicted_labels_test = rf_model.predict(X_test)
print("performance on test set training model only on labelled training data")
print(classification_report(y_test, rf_predicted_labels_test))

# region: spread
# Unlabelled samples get the label -1
X_label_spread = np.concatenate((X_train, X_un))
y_label_spread = np.concatenate((y_train, -1 * np.ones(X_un.shape[0])))

# alpha=0 keeps the initial labels; alpha=1 replaces all of them
lp_model = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2)
lp_model.fit(X_label_spread, y_label_spread)

predicted_labels_unlabelled_set = lp_model.predict(X_un)
predicted_labels_test = lp_model.predict(X_test)
# endregion
print("performance on unlabelled set")
print(classification_report(y_un, predicted_labels_unlabelled_set))

print("performance on test set")
print(classification_report(y_test, predicted_labels_test))
