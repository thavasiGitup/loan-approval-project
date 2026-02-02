from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

# ---------------- APP SETUP ----------------
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

# ---------------- LOAD MODEL ON STARTUP ----------------
@app.on_event("startup")
def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("loan_model.pkl not found in Azure path")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        print("✅ Model loaded successfully")

    except Exception as e:
        print("❌ Model loading failed:", str(e))
        model = None

# ---------------- HEALTH CHECK ----------------
@app.get("/")
def home():
    return {"message": "Loan Approval API running"}

# ---------------- PREDICTION ENDPOINT ----------------
@app.post("/predict")
def predict_loan(data: LoanInput):
    try:
        if model is None:
            raise Exception("Model not loaded")

        # Convert input to numpy array (2D)
        features = np.array([[ 
            float(data.age),
            float(data.income),
            float(data.loan_amount),
            float(data.credit_score)
        ]])

        prediction = model.predict(features)

        result = "Approved" if int(prediction[0]) == 1 else "Rejected"

        return {
            "prediction": int(prediction[0]),
            "result": result
        }

    except Exception as e:
        # IMPORTANT: shows real error in Swagger instead of silent 500
        raise HTTPException(status_code=500, detail=str(e))
