from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0"
)

artifact = joblib.load(
    "fraud_detection_model.pkl"
)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]


class Transaction(BaseModel):

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Hour_sin: float
    Hour_cos: float


@app.get("/")
def home():

    return {
        "message": "Credit Card Fraud Detection API",
        "status": "running"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    data = transaction.model_dump()

    X = pd.DataFrame(
        [data]
    )

    X = X[features]

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    prediction = int(
        probability >= threshold
    )

    return {
        "fraud_probability": round(
            probability,
            4
        ),
        "prediction": prediction,
        "result": (
            "Fraudulent Transaction"
            if prediction == 1
            else "Genuine Transaction"
        ),
        "threshold": threshold
    }