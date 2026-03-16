from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import score_transaction_dict


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time banking fraud scoring API",
    version="1.0.0"
)


class TransactionRequest(BaseModel):
    Time: float = Field(..., example=10000.0)
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = Field(..., example=149.62)


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "endpoint": "/score_transaction"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score_transaction")
def score_transaction(payload: TransactionRequest):
    transaction = payload.model_dump()
    result = score_transaction_dict(transaction)
    return result