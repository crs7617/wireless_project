from fastapi import FastAPI
from data_loader import load_data

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Network Traffic Analysis API"}

@app.get("/data")
def get_data():
    df = load_data()
    if df is not None:
        return df.to_dict(orient="records")
    return {"error": "Failed to load data"}
