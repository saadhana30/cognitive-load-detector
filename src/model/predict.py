import joblib
import numpy as np

model = joblib.load("outputs_model.pkl")

def predict_load(features):
    features = np.array(features).reshape(1, -1)

    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    labels = ["low", "medium", "high"]

    prediction = labels[pred]

    probability = {
        "low": float(probs[0]),
        "medium": float(probs[1]),
        "high": float(probs[2])
    }

    return prediction, probability