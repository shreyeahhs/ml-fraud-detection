from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_proba, predict_labels


class Transaction(BaseModel):
    """
    Simple schema for a single transaction.

    We keep it generic by allowing extra fields,
    because the IEEE-CIS dataset has many columns.
    """
    # You can list some common fields explicitly if you want.
    TransactionAmt: float | None = None
    ProductCD: str | None = None
    card1: int | None = None

    class Config:
        extra = "allow"


class PredictRequest(BaseModel):
    """
    Request body for /predict endpoint.

    We allow either:
    - single: a single transaction dict
    - batch: a list of transactions
    """
    transactions: List[Transaction]


class PredictResponseItem(BaseModel):
    probability: float
    label: int


class PredictResponse(BaseModel):
    predictions: List[PredictResponseItem]


app = FastAPI(
    title="Fraud Detection API",
    description="REST API for IEEE-CIS Fraud Detection model",
    version="1.0.0",
)


@app.get("/")
def read_root():
    """
    Health endpoint to check API is running.
    """
    return {"status": "ok", "message": "Fraud detection API is up"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict fraud for given transactions.

    Example request:
    ----------------
    POST /predict
    {
      "transactions": [
        {
          "TransactionAmt": 100.0,
          "ProductCD": "W",
          "card1": 1000
        },
        {
          "TransactionAmt": 250.5,
          "ProductCD": "C",
          "card1": 2000
        }
      ]
    }

    Example response:
    -----------------
    {
      "predictions": [
        {"probability": 0.12, "label": 0},
        {"probability": 0.78, "label": 1}
      ]
    }
    """
    # Convert list of Transaction objects to list of dict
    data_dicts: List[Dict[str, Any]] = [t.dict() for t in request.transactions]

    probabilities = predict_proba(data_dicts)
    labels = predict_labels(data_dicts)

    response_items = [
        PredictResponseItem(probability=float(p), label=int(l))
        for p, l in zip(probabilities, labels)
    ]

    return PredictResponse(predictions=response_items)
