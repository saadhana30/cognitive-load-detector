import joblib
import pandas as pd
from pathlib import Path

# load model
repo_root = Path(__file__).resolve().parents[2]
model_path = repo_root / "outputs_model.pkl"

model = joblib.load(model_path)

# load some data (for testing)
data_path = repo_root / "data" / "processed" / "labeled_features.csv"
df = pd.read_csv(data_path)

# take sample input (first 5 rows)
X = df.drop(columns=["cognitive_load"])
X = X.select_dtypes(include=["int64", "float64"])

sample = X.head(5)

# predict
predictions = model.predict(sample)

# convert back to labels
mapping = {0: "low", 1: "medium", 2: "high"}
predictions = [mapping[p] for p in predictions]

print("Predictions:")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {pred}")