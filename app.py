from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os

app = FastAPI(title="Loan Approval Prediction API")

# ----------- CORS (FOR REACT) -----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------- MODEL PATH -----------
MODEL_PATH = "/home/site/wwwroot/loan_model.pkl"
model = None

# ----------- INPUT VALIDATION -----------
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

# ----------- LOAD MODEL ONLY -----------
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

# ----------- HEALTH CHECK -----------
@app.get("/")
def home():
    return {"message": "Loan Approval API running"}

# ----------- PREDICTION -----------
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
