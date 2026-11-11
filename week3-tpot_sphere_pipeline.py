# container: scipy
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# The pipeline TPOT 0.12 found for the sphere data, written in plain
# scikit-learn so it runs without TPOT. (TPOT 0.12 exported it with its own
# StackingEstimator; TPOT 1.x has no export(), so print tpot.fitted_pipeline_.)

def generate_sphere_data(num_points, radius, seed=42):
    """Positive points on the surface of a sphere, as in week3-gplearn.py."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, num_points)
    theta = rng.uniform(0, np.pi, num_points)
    x = np.abs(radius * np.sin(theta) * np.cos(phi))
    y = np.abs(radius * np.sin(theta) * np.sin(phi))
    z = np.abs(radius * np.cos(theta))
    return np.column_stack([x, y]), z

features, target = generate_sphere_data(num_points=1000, radius=1.0)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, target, random_state=42)

# region: pipeline
# Step 1: extra trees predict z. Step 2: ElasticNetCV sees x, y and that
# prediction (passthrough=True), and makes the final prediction.
exported_pipeline = StackingRegressor(
    estimators=[("extra_trees",
                 ExtraTreesRegressor(bootstrap=False,
                                     max_features=1.0,
                                     min_samples_leaf=1,
                                     min_samples_split=7,
                                     n_estimators=100,
                                     random_state=42))],
    final_estimator=ElasticNetCV(l1_ratio=0.3,
                                 tol=0.001,
                                 random_state=42),
    passthrough=True,
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
print("Test MSE:", mean_squared_error(testing_target, results))
# endregion
