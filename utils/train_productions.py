# utils/train_productions.py
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from utils.helpers import slugify, error_logs, save_trained_model, save_trained_encoder

TYPE_MODEL_DATA='productions'

def data_processing(df_data):
    # Remove linhas com valores ausentes ou '-'
    df_data = df_data[df_data['amount_liters'].notna()]
    df_data = df_data[df_data['amount_liters'] != '-']

    # Converte tudo para string primeiro
    df_data['amount_liters'] = df_data['amount_liters'].astype(str)
    # Remove os pontos (usados como separador de milhar)
    df_data['amount_liters'] = df_data['amount_liters'].str.replace('.', '', regex=False)
    # Converte para número, tratando valores inválidos como NaN
    df_data['amount_liters'] = pd.to_numeric(df_data['amount_liters'], errors='coerce')

    # Extrai o ano da data
    df_data['year'] = pd.to_datetime(df_data['date']).dt.year

    # Remove valores faltantes
    df_data = df_data.dropna(subset=['amount_liters', 'category', 'name', 'year'])

    df_data = df_data[df_data['name'].str.lower() != 'total']

    return df_data

def filter_data(df_data, wine_name, wine_category, age):
    df_filtered = df_data[(df_data['name'] == wine_name) & (df_data['category'] == wine_category)]

    return df_filtered[df_filtered['year'] >= df_filtered['year'].max() - age]

def split_test_and_train(df_data):
    X = df_data[['name', 'category', 'year']]
    y = df_data['amount_liters']

    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X[['category', 'name']])

    X_final = np.hstack([X_encoded, X[['year']].values])

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, encoder

def train_model(df, wine_name, wine_category):
    df = data_processing(df)

    df_filtered = filter_data(df, wine_name, wine_category, 20)

    X_train, X_test, y_train, y_test, encoder = split_test_and_train(df_filtered)

    # Linear regression
    model = LinearRegression()

    y_train_log = np.log1p(y_train)

    model.fit(X_train, y_train_log)

    y_pred_log = model.predict(X_test)

    y_pred_litros = np.expm1(y_pred_log)

    y_test_litros = y_test.values

    error_logs(y_test_litros, y_pred_litros)

    # Save model trained
    filename = slugify(f'{wine_category} {wine_name}')

    save_trained_model(model, TYPE_MODEL_DATA, filename)

    save_trained_encoder(encoder, TYPE_MODEL_DATA, filename)