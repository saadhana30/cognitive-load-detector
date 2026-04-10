from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def load_dataset(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def prepare_data(df: pd.DataFrame):
    if "cognitive_load" not in df.columns:
        raise ValueError("Target column 'cognitive_load' not found.")

    # Separate features and target
    X = df.drop(columns=["cognitive_load"])

    # ✅ IMPORTANT FIX: keep only numeric columns
    X = X.select_dtypes(include=["int64", "float64"])

    y = df["cognitive_load"].map({"low": 0, "medium": 1, "high": 2})

    if y.isna().any():
        raise ValueError("Invalid labels found in cognitive_load column.")

    return X, y.astype(int)


def build_hybrid_model():
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf)],
        voting="soft"
    )

    return model


def main():
    repo_root = Path(__file__).resolve().parents[2]

    data_path = repo_root / "data" / "processed" / "labeled_features.csv"
    model_path = repo_root / "outputs_model.pkl"

    df = load_dataset(data_path)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_hybrid_model()

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")

    # Save model
    joblib.dump(model, model_path)
    print(f"✅ Model saved at: {model_path}")


if __name__ == "__main__":
    main()