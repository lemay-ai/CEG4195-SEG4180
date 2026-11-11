# container: scipy
"""Track a model with MLflow: log a run, register the model, give the new version an alias,
then load the model back by that alias.

    python week11-mlflow.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db    # then open http://127.0.0.1:5000

MLflow keeps the runs and the model registry in mlflow.db, and the model files in mlruns/,
both in the current folder. Each run adds a new version of "iris-classifier" and moves the
@champion alias to it. On Colab, week11-mlflow_on_collab.ipynb runs the same code, then gives
the MLflow UI a public address through a tunnel.
"""
import mlflow
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")  # a local database for the runs and the registry
mlflow.set_experiment("iris")

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# skops saves the model and loads back only the types listed as trusted: here, a forest's trees
TRUSTED = ["sklearn.tree._tree.Tree"]

# region: registry
params = {"n_estimators": 10, "random_state": 42}
with mlflow.start_run():
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    # Save with skops, then register a new version
    info = mlflow.sklearn.log_model(
        model, name="model",
        input_example=X_train[:3],
        registered_model_name="iris-classifier",
        skops_trusted_types=TRUSTED)
# Point @champion at this version, load by alias
version = info.registered_model_version
MlflowClient().set_registered_model_alias(
    "iris-classifier", "champion", version)
uri = "models:/iris-classifier@champion"
champion = mlflow.sklearn.load_model(uri)
print("Version", version, champion.score(X_test, y_test))
# endregion
