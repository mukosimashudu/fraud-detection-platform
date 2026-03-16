import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.preprocessing import create_features


THRESHOLD_PATH = "models/fraud_threshold.pkl"


def load_model_and_threshold():
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return model, threshold


def score_transaction_dict(transaction: dict) -> dict:
    model, threshold = load_model_and_threshold()

    df = pd.DataFrame([transaction])

    df = create_features(df)

    probs = model.predict_proba(df)[:, 1]
    prob = float(probs[0])

    predicted_class = int(prob > threshold)

    if prob >= 0.85:
        risk_level = "Critical"
    elif prob >= threshold:
        risk_level = "High"
    elif prob >= max(threshold * 0.5, 0.10):
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "fraud_probability": round(prob, 6),
        "threshold": round(float(threshold), 6),
        "predicted_class": predicted_class,
        "risk_level": risk_level
    }