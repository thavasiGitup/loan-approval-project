from fastapi import FastAPI, HTTPException
import pickle
import os
import numpy as np

app = FastAPI()

MODEL_PATH = "/home/site/wwwroot/loan_model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model file not found")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Loan Approval API running"}

@app.post("/predict")
def predict(data: dict):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    features = np.array([list(data.values())])
    prediction = model.predict(features)
    return {"loan_approval": int(prediction[0])}
