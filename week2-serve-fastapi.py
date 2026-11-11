# container: scipy
"""Serve a scikit-learn model behind an API with FastAPI: POST /predict.

    python week2-serve-fastapi.py           # self-test: one request through FastAPI's TestClient
    python week2-serve-fastapi.py --serve   # serve on port 8000 (try http://localhost:8000/docs)

The first run trains a small classifier on the iris flowers and saves it to
iris-model.joblib. The API only loads that file, as it would in production,
where the trained model is copied into the image.

In a container (fastapi, uvicorn and scikit-learn are in the course's scipy image):

    docker run --rm -p 8000:8000 -v "$PWD":/home/jovyan/work -w /home/jovyan/work \
        course-code-scipy:2026-11 python week2-serve-fastapi.py --serve
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
        -d '{"sepal_length": 6.3, "sepal_width": 2.9, "petal_length": 5.6, "petal_width": 1.8}'
"""

import sys
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

MODEL_FILE = Path("iris-model.joblib")

if not MODEL_FILE.exists():  # train once and save; the API below only loads the file
    iris = load_iris()
    clf = LogisticRegression(max_iter=1000).fit(iris.data, iris.target)
    joblib.dump({"model": clf, "classes": list(iris.target_names)}, MODEL_FILE)

# region: serve
from fastapi import FastAPI
from pydantic import BaseModel

saved = joblib.load(MODEL_FILE)  # once
app = FastAPI()

class Flower(BaseModel):  # the request body
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(f: Flower):
    x = [[f.sepal_length, f.sepal_width,
          f.petal_length, f.petal_width]]
    p = saved["model"].predict_proba(x)[0]
    i = int(p.argmax())
    return {"species": saved["classes"][i],
            "probability": round(p[i], 3)}
# endregion

if __name__ == "__main__":
    if "--serve" in sys.argv:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # region: test
        from fastapi.testclient import TestClient

        client = TestClient(app)             # calls the app without starting a server
        r = client.post("/predict", json={"sepal_length": 6.3, "sepal_width": 2.9,
                                          "petal_length": 5.6, "petal_width": 1.8})
        print(r.status_code, r.json())
        bad = client.post("/predict", json={"sepal_length": "tall"})
        print(bad.status_code, "for a bad request")
        # endregion
        assert r.status_code == 200 and r.json()["species"] == "virginica", r.text
        assert bad.status_code == 422, bad.text
