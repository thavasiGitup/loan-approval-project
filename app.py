from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def home():
    return {"status": "FastAPI running"}

@app.get("/health")
def health():
    return {"ok": True}
