# app/main.py
from fastapi import APIRouter
from utils.helpers import load_data, load_productions_categories
from utils.train_productions import train_model

router = APIRouter()

@router.post("/training")
async def training(data: dict):
    df = load_data(data["type"])

    if data["type"] == 'productions':
        dl = load_productions_categories()

        for categoria in dl['categories']:
            for nome in dl['names_by_category'][categoria]:
                train_model(df, nome, categoria)

    return {
        "training": True
    }