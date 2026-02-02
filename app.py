from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="Loan Approval Prediction API")

# ---------------- MODEL PATH ----------------
MODEL_PATH = "/home/site/wwwroot/loan_model.pkl"
model = None

# ---------------- INPUT SCHEMA (ALL 9 FEATURES) ----------------
class LoanInput(BaseModel):
    age: int
    income: float
    loan_amount: float
    credit_score: int
    gender: int           # Male=1, Female=0
    married: int          # Yes=1, No=0
    dependents: int       # 0,1,2,3
    education: int        # Graduate=1, Not Graduate=0
    self_employed: int    # Yes=1, No=0

# ---------------- LOAD MODEL ----------------
@app.on_event("startup")
def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("loan_model.pkl not found")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        print("✅ Model loaded successfully")

    except Exception as e:
        print("❌ Model loading failed:", e)
        model = None

# ---------------- HOME ----------------
@app.get("/")
def home():
    return {"message": "Loan Approval API running"}

# ---------------- PREDICTION ----------------
@app.post("/predict")
def predict_loan(data: LoanInput):
    try:
        if model is None:
            raise Exception("Model not loaded")

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

        prediction = model.predict(features)[0]

        return {
            "prediction": int(prediction),
            "result": "Approved" if prediction == 1 else "Rejected"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
