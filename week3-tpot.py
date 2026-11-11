# container: scipy
import os

import matplotlib.pyplot as plt
import numpy as np
# region: imports
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# endregion

# Written for TPOT 1.x (pip install tpot; needs Python 3.10 to 3.13).


def generate_sphere_data(num_points, radius, seed=42):
    """Positive points on the surface of a sphere, as in week3-gplearn.py."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, num_points)  # Azimuthal angle
    theta = rng.uniform(0, np.pi, num_points)  # Polar angle
    x = np.abs(radius * np.sin(theta) * np.cos(phi))
    y = np.abs(radius * np.sin(theta) * np.sin(phi))
    z = np.abs(radius * np.cos(theta))
    return np.column_stack([x, y]), z


# region: search
# Inputs: positive x and y on the sphere. Output: z.
X, y = generate_sphere_data(num_points=1000, radius=1.0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.75, test_size=0.25, random_state=42)

# Configure TPOT (processes=False: threads, so no __main__ guard)
tpot = TPOTRegressor(search_space="linear",
                     generations=5,
                     population_size=50,
                     max_time_mins=10,
                     processes=False,
                     verbose=2, random_state=42)
# endregion

# The smoke test (make code-test) runs a tiny search: 2 generations of 4 pipelines,
# each scored with 2-fold cross-validation (TPOTRegressor's default is cv=10).
if os.environ.get("COURSE_SMOKE"):
    tpot = TPOTRegressor(search_space="linear", generations=2, population_size=4, cv=2,
                         max_time_mins=10, processes=False, verbose=2, random_state=42)

# region: fit
# Perform the search
tpot.fit(X_train, y_train)
# Show the best pipeline (a scikit-learn object; TPOT 1.x has no export())
print(tpot.fitted_pipeline_)
# Evaluate the pipeline
predicted_y = tpot.predict(X_test)
print("Test MSE:", mean_squared_error(y_test, predicted_y))
# endregion

# Plot predicted vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicted_y, alpha=0.6, label="Predicted vs True")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Ideal")
plt.xlabel("True Z Values")
plt.ylabel("Predicted Z Values")
plt.title("True vs Predicted Z Values on the Sphere using TPOT pipeline")
plt.legend()
plt.show()
