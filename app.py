from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="Loan Approval Prediction API")

# ---------------- MODEL PATH ----------------
MODEL_PATH = "/home/site/wwwroot/loan_model.pkl"
model = None

# ---------------- INPUT SCHEMA ----------------
class LoanInput(BaseModel):
    age: int
    income: float
    loan_amount: float
    credit_score: int

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
        print("❌ Model loading error:", e)
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

        # --------------------------------------------------
        # IMPORTANT: 9 FEATURES IN THE SAME ORDER AS TRAINING
        # --------------------------------------------------

        features = np.array([[ 
            float(data.age),           # 1
            float(data.income),        # 2
            float(data.loan_amount),   # 3
            float(data.credit_score),  # 4

            1,  # 5 gender (default: Male=1)
            1,  # 6 married (Yes=1)
            0,  # 7 dependents (0)
            1,  # 8 education (Graduate=1)
            0   # 9 self_employed (No=0)
        ]])

        prediction = model.predict(features)[0]

        return {
            "prediction": int(prediction),
            "result": "Approved" if prediction == 1 else "Rejected"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
