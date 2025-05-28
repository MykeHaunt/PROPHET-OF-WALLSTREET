import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import IsolationForest

def run_isolation_forest(df, config):
    logging.info("Starting Isolation Forest anomaly detection...")
    df_if = df.copy()
    df_if['Date'] = pd.to_datetime(df_if['Date'])
    df_if.sort_values(by='Date', inplace=True)
    df_if.set_index('Date', inplace=True)
    features = df_if[['Close', 'Volume']].fillna(0).values
    model = IsolationForest(n_estimators=config['isolation']['n_estimators'],
                            contamination=config['isolation']['contamination'],
                            random_state=42)
    preds = model.fit_predict(features)
    df_if['anomaly'] = (preds == -1)
    plt.figure(figsize=(10,6))
    plt.plot(df_if.index, df_if['Close'], label='Close Price')
    anomalies = df_if[df_if['anomaly']]
    plt.scatter(anomalies.index, anomalies['Close'], color='red', label='Anomaly')
    plt.xlabel('Date'); plt.ylabel('Price'); plt.title('Isolation Forest Anomalies')
    plt.legend(); plt.tight_layout()
    plt.savefig(config['paths']['output_dir'] + '/isolation_forest.png')
    plt.close()
    logging.info("Isolation Forest detection completed. Plot saved.")
    return df_if.reset_index()[['Date','Close','Volume','anomaly']]
