import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from utils.preprocessing import create_sequences
from utils.evaluation import evaluate_forecast

def run_lstm(df, config):
    logging.info("Starting LSTM forecasting...")
    series = df[['Date', 'Close']].copy()
    series['Close'] = series['Close'].astype(float)
    series['Date'] = pd.to_datetime(series['Date'])
    series.set_index('Date', inplace=True)
    data = series[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    window_size = config['lstm']['window_size']
    X, y = create_sequences(data_scaled, window_size)
    test_size = config['lstm'].get('test_size', 0.2)
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    model = Sequential([
        LSTM(config['lstm']['units'], input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info(f"LSTM model compiled (units={config['lstm']['units']})")
    model.fit(X_train, y_train, epochs=config['lstm']['epochs'], 
              batch_size=config['lstm']['batch_size'], verbose=0)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1,1))
    date_index = series.index[window_size+split_index:]
    results = pd.DataFrame({'ds': date_index,
                            'actual': actual.flatten(),
                            'predicted': predictions.flatten()})
    plt.figure(figsize=(10,6))
    plt.plot(results['ds'], results['actual'], label='Actual')
    plt.plot(results['ds'], results['predicted'], label='Predicted')
    plt.xlabel('Date'); plt.ylabel('Price'); plt.title('LSTM Forecast')
    plt.legend(); plt.tight_layout()
    plt.savefig(config['paths']['output_dir'] + '/lstm_forecast.png')
    plt.close()
    logging.info("LSTM forecasting completed. Plot saved.")
    return results
