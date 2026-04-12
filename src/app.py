from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import joblib
import pandas as pd
from pathlib import Path
import os
import tempfile

from src.database import init_db, save_feedback
from src.speech_to_text import audio_to_text
from src.generator_ai import generate_summary, generate_quiz

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ INIT DB
init_db()

# ✅ PATH
repo_root = Path(__file__).resolve().parents[1]

# ✅ SAFE MODEL LOAD
model = None
model_path = repo_root / "outputs_model.pkl"

try:
    if model_path.exists():
        model = joblib.load(model_path)
        print("Model loaded successfully")
    else:
        print("WARNING: Model file missing!")
except Exception as e:
    print("Model load error:", e)


# ✅ FEATURE COLUMNS (SAFE)
feature_cols = []
try:
    df_temp = pd.read_csv(repo_root / "data/processed/labeled_features.csv")
    feature_cols = df_temp.drop(columns=["cognitive_load"]) \
                          .select_dtypes(include=["int64", "float64"]) \
                          .columns
except Exception as e:
    print("Feature load error:", e)


mapping = {0: "low", 1: "medium", 2: "high"}


@app.get("/")
def home():
    return {"message": "Backend Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("Received file")

        # SAVE FILE
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            shutil.copyfileobj(file.file, temp)
            file_path = temp.name

        print("Audio saved")

        # AUDIO → TEXT
        text = audio_to_text(file_path)
        print("Converted to text")

        # CLEAN TEMP FILE
        os.remove(file_path)

        # SAFE: if model missing
        if model is None or len(feature_cols) == 0:
            return {
                "prediction": "medium",
                "probability": {
                    "low": 0.2,
                    "medium": 0.6,
                    "high": 0.2
                },
                "summary": generate_summary(text),
                "quiz": generate_quiz(text)
            }

        # CREATE FEATURES
        features = {col: 0.0 for col in feature_cols}

        features["speech_rate"] = 170
        features["pause_frequency"] = 1
        features["mean_pitch"] = 300
        features["word_count"] = len(text.split()) if text else 0

        df = pd.DataFrame([features])
        df = df[feature_cols]

        # PREDICT
        pred = model.predict(df)[0]
        probs = model.predict_proba(df)[0]

        label = mapping[int(pred)]

        return {
            "prediction": label,
            "probability": {
                "low": float(probs[0]),
                "medium": float(probs[1]),
                "high": float(probs[2])
            },
            "summary": generate_summary(text),
            "quiz": generate_quiz(text)
        }

    except Exception as e:
        print("ERROR:", e)
        return {"error": str(e)}


@app.post("/feedback")
def feedback(data: dict):
    save_feedback(data.get("prediction"), data.get("feedback"))
    return {"message": "Feedback saved"}