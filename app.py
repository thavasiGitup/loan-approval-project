from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="Loan Approval Prediction API")

# -------- MODEL PATH --------
MODEL_PATH = "/home/site/wwwroot/loan_model.pkl"
model = None

# -------- INPUT SCHEMA --------
class LoanInput(BaseModel):
    age: int
    income: float
    loan_amount: float
    credit_score: int

# -------- LOAD MODEL ON STARTUP --------
@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model file not found in Azure")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# -------- HEALTH CHECK --------
@app.get("/")
def home():
    return {"message": "Loan Approval API running"}

# -------- PREDICTION ENDPOINT --------
@app.post("/predict")
def predict_loan(data: LoanInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = np.array([[ 
        data.age,
        data.income,
        data.loan_amount,
        data.credit_score
    ]])

    prediction = model.predict(features)[0]

    result = "Approved" if prediction == 1 else "Rejected"

    return {
        "prediction": int(prediction),
        "result": result
    }
