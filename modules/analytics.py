# order_processing_system/analytics.py

import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import statsmodels.api as sm
from prophet import Prophet

def train_lstm_model(data, feature='quantity'):
    """
    Trains an LSTM model on the provided sales data.
    """
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature].values.reshape(-1, 1))

    # Prepare input/output sequences for LSTM
    x_train, y_train = [], []
    sequence_length = 30  # last 30 days data

    for i in range(sequence_length, len(scaled_data)):
        x_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # [samples, time steps, features]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=20)

    logging.info("LSTM model trained successfully.")
    return model, scaler

def lstm_predict_next_30_days(data, model, scaler):
    """
    Uses the trained LSTM model to predict the next 30 days of sales.
    """
    last_30_days = data[-30:].values.reshape(-1, 1)
    scaled_last_30_days = scaler.transform(last_30_days)

    # Predict the next 30 days
    x_input = np.reshape(scaled_last_30_days, (1, scaled_last_30_days.shape[0], 1))
    predictions = []

    for _ in range(30):
        pred = model.predict(x_input)
        predictions.append(pred[0, 0])
        x_input = np.append(x_input[:, 1:, :], [[pred]], axis=1)

    predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    logging.info("LSTM predictions for next 30 days generated.")
    return predicted_values

def train_arima_model(data, feature='quantity'):
    """
    Trains an ARIMA model on the provided sales data.
    """
    # Fit ARIMA model
    model = sm.tsa.ARIMA(data[feature].values, order=(5, 1, 0))  # Adjust order for ARIMA
    model_fit = model.fit(disp=0)
    logging.info("ARIMA model trained successfully.")
    return model_fit

def arima_predict_next_30_days(model_fit):
    """
    Uses the trained ARIMA model to predict the next 30 days of sales.
    """
    # Predict next 30 days
    forecast = model_fit.forecast(steps=30)[0]
    logging.info("ARIMA predictions for next 30 days generated.")
    return forecast

def train_prophet_model(data):
    """
    Trains a Prophet model on the provided sales data.
    """
    # Prepare data for Prophet
    prophet_data = data.rename(columns={'date': 'ds', 'quantity': 'y'})

    # Initialize Prophet model
    model = Prophet()
    model.fit(prophet_data)

    logging.info("Prophet model trained successfully.")
    return model

def prophet_predict_next_30_days(model):
    """
    Uses the trained Prophet model to predict the next 30 days of sales.
    """
    # Predict future values for the next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    predictions = forecast[['ds', 'yhat']].tail(30)  # Return the last 30 days of prediction
    logging.info("Prophet predictions for next 30 days generated.")
    return predictions
