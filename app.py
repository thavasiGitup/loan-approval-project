from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.get("/")
def home():
    return {"message": "Loan Approval Prediction API"}

@app.post("/predict")
def predict(data: dict):
    features = np.array([[ 
        data["Applicant_Income"],
        data["Coapplicant_Income"],
        data["Loan_Amount"],
        data["Loan_Term"],
        data["Credit_History"]
    ]])

    prediction = model.predict(features)

    result = "Approved" if prediction[0] == 1 else "Rejected"
    return {"Loan_Status": result}
