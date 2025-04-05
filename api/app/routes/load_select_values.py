# app/main.py
from fastapi import APIRouter
from utils.helpers import load_productions_categories

router = APIRouter()

@router.get("/options/productions")
async def load_select_values():
    return load_productions_categories()