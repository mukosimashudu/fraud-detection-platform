import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE

from src.database import create_database, read_transactions
from src.preprocessing import split_data
from src.config import MODEL_PATH, THRESHOLD_PATH, METRICS_PATH


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a fraud model using probability-based metrics
    and find the optimal threshold using the PR curve.
    """

    probs = model.predict_proba(X_test)[:, 1]

    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, probs)

    # thresholds has length n-1 while precision/recall usually has length n
    # align them safely
    if len(thresholds) == 0:
        best_threshold = 0.5
    else:
        f1_scores = 2 * precision_curve[:-1] * recall_curve[:-1] / (
            precision_curve[:-1] + recall_curve[:-1] + 1e-10
        )
        best_index = np.argmax(f1_scores)
        best_threshold = thresholds[best_index]

    preds = (probs >= best_threshold).astype(int)

    precision_val = precision_score(y_test, preds, zero_division=0)
    recall_val = recall_score(y_test, preds, zero_division=0)
    f1_val = f1_score(y_test, preds, zero_division=0)
    roc_val = roc_auc_score(y_test, probs)
    pr_val = average_precision_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print(f"Optimal threshold: {best_threshold:.6f}")
    print(f"Precision: {precision_val:.4f}")
    print(f"Recall: {recall_val:.4f}")
    print(f"F1 Score: {f1_val:.4f}")
    print(f"ROC-AUC: {roc_val:.4f}")
    print(f"PR-AUC: {pr_val:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1_val,
        "roc_auc": roc_val,
        "pr_auc": pr_val,
        "confusion_matrix": cm,
        "threshold": float(best_threshold)
    }


def build_models():
    """
    Define candidate fraud detection models.
    """

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=8000,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "XGBoost": CalibratedClassifierCV(
            estimator=XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=42
            ),
            method="isotonic",
            cv=3
        )
    }

    return models


def train_models():
    """
    Train fraud models, compare them, log to MLflow,
    and save the best model + threshold.
    """

    print("Creating database...")
    create_database()

    print("Loading transactions...")
    df = read_transactions()

    print("Preparing train/test split...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Handling class imbalance with SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    models = build_models()

    best_model = None
    best_model_name = None
    best_threshold = 0.5
    best_score = -1.0

    results = []

    mlflow.set_experiment("fraud-detection")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\nTraining {name}...")

            model.fit(X_train_res, y_train_res)

            metrics = evaluate_model(model, X_test, y_test)

            # log metrics
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("f1", metrics["f1"])
            mlflow.log_metric("roc_auc", metrics["roc_auc"])
            mlflow.log_metric("pr_auc", metrics["pr_auc"])
            mlflow.log_metric("threshold", metrics["threshold"])

            # log useful params
            mlflow.log_param("model_name", name)

            # log model
            mlflow.sklearn.log_model(sk_model=model, name=name)

            results.append({
                "model": name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "threshold": metrics["threshold"]
            })

            # choose best model by PR-AUC
            if metrics["pr_auc"] > best_score:
                best_score = metrics["pr_auc"]
                best_model = model
                best_model_name = name
                best_threshold = metrics["threshold"]

    # save leaderboard
    results_df = pd.DataFrame(results).sort_values(by="pr_auc", ascending=False)
    results_df.to_csv(METRICS_PATH, index=False)

    print("\nModel comparison:")
    print(results_df)

    # save best model and threshold
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(best_threshold, THRESHOLD_PATH)

    print(f"\nBest model: {best_model_name}")
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"Optimal threshold saved to: {THRESHOLD_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train_models()