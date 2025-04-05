# shared/config.py
from dotenv import load_dotenv
import os

load_dotenv(override=True)

API_URL=os.getenv("API_URL")
API_TOKEN=os.getenv("API_TOKEN")

WINE_NAME=os.getenv("WINE_NAME")
WINE_CATEGORY=os.getenv("WINE_CATEGORY")