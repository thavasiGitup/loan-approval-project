from fastapi import FastAPI
import pickle
import os
import numpy as np

app = FastAPI()

MODEL_PATH = "/home/site/wwwroot/loan_model.pkl"

@app.on_event("startup")
def load_model():
    global model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Loan Approval ML API is running"}

@app.post("/predict")
def predict(data: dict):
    features = np.array([list(data.values())])
    prediction = model.predict(features)
    return {"loan_approval": int(prediction[0])}
