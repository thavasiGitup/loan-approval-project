from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os

app = FastAPI(title="Loan Approval Prediction API")

# ---------------- PATHS ----------------
MODEL_PATH = "/home/site/wwwroot/loan_model.pkl"
SCALER_PATH = "/home/site/wwwroot/scaler.pkl"

model = None
scaler = None

# ---------------- INPUT VALIDATION ----------------
class LoanInput(BaseModel):
    age: int = Field(..., gt=17, lt=100)
    income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=900)

    gender: int = Field(..., ge=0, le=1)
    married: int = Field(..., ge=0, le=1)
    dependents: int = Field(..., ge=0, le=3)
    education: int = Field(..., ge=0, le=1)
    self_employed: int = Field(..., ge=0, le=1)

# ---------------- LOAD MODEL & SCALER ----------------
@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("loan_model.pkl not found")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("scaler.pkl not found")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        print("✅ Model and scaler loaded successfully")

    except Exception as e:
        print("❌ Loading failed:", e)
        model = None
        scaler = None

# ---------------- HOME ----------------
@app.get("/")
def home():
    return {"message": "Loan Approval API running"}

# ---------------- PREDICTION ----------------
@app.post("/predict")
def predict_loan(data: LoanInput):
    try:
        if model is None or scaler is None:
            raise Exception("Model or scaler not loaded")

        # Raw features (same order as training)
        features = np.array([[ 
            data.age,
            data.income,
            data.loan_amount,
            data.credit_score,
            data.gender,
            data.married,
            data.dependents,
            data.education,
            data.self_employed
        ]])

        # APPLY SCALING (IMPORTANT)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        return {
            "prediction": int(prediction),
            "result": "Approved" if prediction == 1 else "Rejected"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
