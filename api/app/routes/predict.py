# app/main.py
from fastapi import APIRouter
from utils.helpers import load_prediction

router = APIRouter()

@router.post("/predict")
async def predict(data: dict):
    y_pred_litros = load_prediction(data["type"], data["category"], data["name"], data["year"])

    return {
        "prediction": f"{y_pred_litros[0]:,.0f}"
    }