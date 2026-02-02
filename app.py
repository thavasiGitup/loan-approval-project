from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "FastAPI app is running on Azure"}
