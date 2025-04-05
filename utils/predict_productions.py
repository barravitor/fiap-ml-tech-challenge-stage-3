# utils/predict_productions.py
import pandas as pd
import numpy as np

from utils.helpers import slugify, load_trained_model, load_trained_encoder, forecast

def load_prediction(type_model_data, wine_categoy, wine_name, wine_year):
    filename = slugify(f'{wine_categoy} {wine_name}')

    # Load model
    model = load_trained_model(type_model_data, filename)

    # Load encoder
    encoder = load_trained_encoder(type_model_data, filename)

    new_test_model = pd.DataFrame({
        'name': [wine_name],
        'category': [wine_categoy],
        'year': [wine_year]
    })

    # Encoding with the trained encoder
    encoded_data = encoder.transform(new_test_model[['category', 'name']])

    # Merge with year column
    X_novo = np.hstack([encoded_data, new_test_model[['year']].values])

    # Forecast
    y_pred = model.predict(X_novo)

    # Convert log to Y
    y_pred_litros = np.expm1(y_pred)

    # Display the forecast
    forecast(new_test_model, y_pred_litros)

    return y_pred_litros