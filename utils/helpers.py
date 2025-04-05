# utils/helpers.py
import os
import requests
from io import StringIO
import pandas as pd
import numpy as np
import joblib

from utils.config import API_URL, API_TOKEN

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

def load_data(type):
    if os.path.exists(f'./data/data_embrapa_{type}.csv'):
        return pd.read_csv(f'./data/data_embrapa_{type}.csv')

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Accept": "text/csv"
    }

    response = requests.get(f'{API_URL}/embrapa/{type}', headers=headers)

    if response.status_code == 200:
        csv_data = StringIO(response.text)

        df = pd.read_csv(csv_data)

        df.to_csv(f'./data/data_embrapa_{type}.csv', index=False)

        return df
    else:
        print(f"Error on request: {response.status_code}")
        print(response.text)

def slugify(text):
    """
    Turns text into a slug: lowercase, without spaces or dashes.
    Example: 'Table Wine - White' → 'white_table_wine'
    """
    return text.lower().replace(" ", "_").replace("-", "_")

def forecast(model, y_pred_litros):
    """
    Display the forecast
    Example: 'Forecast for 2024 | (Branco, VINHO FINO DE MESA): 10.000 liters'
    """
    print(f"Forecast for {model['year'].iloc[0]} | ({model['name'].iloc[0]}, {model['category'].iloc[0]}): {y_pred_litros[0]:,.0f} liters")

def error_logs(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    print(f"\nMSE: {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAPE: {mape:.2f}%")

def save_trained_model(model, type, filename): 
    joblib.dump(model, f'./models/{type}/{filename}_model_trained.pkl')

def save_trained_encoder(encoder, type, filename): 
    joblib.dump(encoder, f'./encoders/{type}/{filename}_encoder_trained.pkl')

def load_trained_model(type, filename):
    return joblib.load(f'./models/{type}/{filename}_model_trained.pkl')

def load_trained_encoder(type, filename):
    return joblib.load(f'./encoders/{type}/{filename}_encoder_trained.pkl')