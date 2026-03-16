from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

DATABASES_DIR = BASE_DIR / "databases"
DB_PATH = DATABASES_DIR / "fraud_detection.db"
MLFLOW_DB_PATH = DATABASES_DIR / "mlflow.db"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
THRESHOLD_PATH = MODELS_DIR / "fraud_threshold.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.csv"
FORECAST_PATH = MODELS_DIR / "fraud_forecast.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALID_SIZE = 0.25

TARGET_COL = "Class"

DATABASES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)