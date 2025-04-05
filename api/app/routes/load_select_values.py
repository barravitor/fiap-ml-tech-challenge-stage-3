# app/main.py
from fastapi import APIRouter
from utils.helpers import load_data

router = APIRouter()

@router.post("/options")
async def load_select_values(data: dict):
    df = load_data(data["type"])

    df = df[df['amount_liters'] != '-']

    df['name'] = df['name'].str.strip()
    df['category'] = df['category'].str.strip()

    df = df[df['name'].str.lower() != 'total']

    categories = sorted(df['category'].unique().tolist())

    return {
        "categories": categories,
        "names_by_category": df[['name', 'category']]
            .drop_duplicates()
            .groupby('category')['name']
            .apply(list)
            .to_dict(),
    }