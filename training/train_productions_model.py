from utils.helpers import load_data
from utils.train_productions import train_model
from utils.config import WINE_CATEGORY, WINE_NAME

TYPE_MODEL_DATA='productions'

df = load_data(TYPE_MODEL_DATA)
train_model(df, WINE_NAME, WINE_CATEGORY)