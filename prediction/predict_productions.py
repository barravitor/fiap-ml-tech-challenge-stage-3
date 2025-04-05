import numpy as np
import pandas as pd

from utils.helpers import slugify, forecast, load_trained_model, load_trained_encoder
from utils.config import WINE_CATEGORY, WINE_NAME

filename = slugify(f'{WINE_CATEGORY} {WINE_NAME}')

# Load model
model = load_trained_model(filename)

# Load encoder
encoder = load_trained_encoder(filename)

new_test_model = pd.DataFrame({
    'name': [WINE_NAME],
    'category': [WINE_CATEGORY],
    'year': [2025]
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