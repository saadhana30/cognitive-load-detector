if not (repo_root / "outputs_model.pkl").exists():
    import subprocess
    subprocess.run(["python", "src/model/train.py"])
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import joblib
import pandas as pd
from pathlib import Path
import os

from src.database import init_db, save_feedback
from src.speech_to_text import audio_to_text

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

# ✅ PATHS
repo_root = Path(__file__).resolve().parents[1]

# ✅ LOAD MODEL ONCE (FAST)
model = joblib.load(repo_root / "outputs_model.pkl")

# ✅ LOAD FEATURE COLUMNS ONCE (FAST)
train_df = pd.read_csv(repo_root / "data/processed/labeled_features.csv")
feature_cols = train_df.drop(columns=["cognitive_load"]) \
                       .select_dtypes(include=["int64", "float64"]) \
                       .columns

mapping = {0: "low", 1: "medium", 2: "high"}


@app.get("/")
def home():
    return {"message": "API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # --------------------------
        # SAVE FILE
        # --------------------------
        file_path = f"temp_{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --------------------------
        # AUDIO → TEXT (FAST LIMITED)
        # --------------------------
        try:
            text = audio_to_text(file_path)
        except:
            text = ""

        # --------------------------
        # FEATURE CREATION
        # --------------------------
        features = {col: 0.0 for col in feature_cols}

        features["speech_rate"] = 170
        features["pause_frequency"] = 1
        features["mean_pitch"] = 300
        features["word_count"] = len(text.split()) if text else 0

        df = pd.DataFrame([features])
        df = df[feature_cols]

        # --------------------------
        # PREDICT
        # --------------------------
        pred = model.predict(df)[0]
        probs = model.predict_proba(df)[0]

        label = mapping[int(pred)]

        # --------------------------
        # 🔥 BETTER SUMMARY
        # --------------------------
        if text:
            sentences = text.split(".")
            summary = ". ".join(sentences[:2])[:250]
        else:
            summary = "No clear speech detected."

        # --------------------------
        # 🔥 BETTER QUIZ
        # --------------------------
        quiz = []
        if text:
            words = text.split()
            topic = " ".join(words[:5])

            quiz = [
                f"What is the main concept discussed about '{topic}'?",
                f"Explain the idea behind '{topic}'.",
                f"What conclusion can you draw from this lecture?"
            ]
        else:
            quiz = ["No quiz generated"]

        # --------------------------
        # CLEAN FILE
        # --------------------------
        os.remove(file_path)

        return {
            "prediction": label,
            "probability": {
                "low": float(probs[0]),
                "medium": float(probs[1]),
                "high": float(probs[2])
            },
            "summary": summary,
            "quiz": quiz
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/feedback")
def feedback(data: dict):
    save_feedback(data.get("prediction"), data.get("feedback"))
    return {"message": "Feedback saved"}