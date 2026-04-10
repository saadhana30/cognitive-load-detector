from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI()

# load model
repo_root = Path(__file__).resolve().parents[1]
model = joblib.load(repo_root / "outputs_model.pkl")

# label mapping
mapping = {0: "low", 1: "medium", 2: "high"}


@app.get("/")
def home():
    return {"message": "Cognitive Load API running"}


@app.post("/predict")
def predict(data: dict):
    try:
        # load training data to match columns
        train_df = pd.read_csv(repo_root / "data" / "processed" / "labeled_features.csv")

        train_columns = train_df.drop(columns=["cognitive_load"]) \
                                .select_dtypes(include=["int64", "float64"]) \
                                .columns

        # create input dataframe
        df = pd.DataFrame(columns=train_columns)

        for col in train_columns:
            df.loc[0, col] = float(data.get(col, 0))

        # ensure numeric
        df = df.astype(float)

        # prediction
        pred = model.predict(df)[0]
        probs = model.predict_proba(df)[0]

        label = mapping[pred]

        response = {
            "prediction": label,
            "probability": {
                "low": float(probs[0]),
                "medium": float(probs[1]),
                "high": float(probs[2])
            }
        }

        # 🔥 SAFE summary + quiz (no crash)
        if label == "high":
            response["summary"] = "This lecture contains complex concepts and may require careful review."
            response["quiz"] = [
                "What is the main topic of the lecture?",
                "Explain one key concept discussed.",
                "What are the important takeaways?"
            ]

        return response

    except Exception as e:
        return {"error": str(e)}