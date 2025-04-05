# app/main.py
from fastapi import APIRouter
from utils.helpers import load_data
from utils.predict_productions import load_prediction
from utils.train_productions import train_model

router = APIRouter()

@router.post("/predict")
async def predict(data: dict):
    df = load_data(data["type"])
    train_model(df, data["name"], data["category"])

    y_pred_litros = load_prediction(data["type"], data["category"], data["name"], data["year"])

    return {
        "prediction": f"{y_pred_litros[0]:,.0f}"
    }