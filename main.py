import os
import argparse
import logging
import pandas as pd
import yaml

from utils.preprocessing import load_data
from utils.evaluation import evaluate_forecast
from models.prophet_forecast import run_prophet
from models.lstm_forecast import run_lstm
from models.isolation_forest import run_isolation_forest

def setup_logging(log_file):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

def main():
    parser = argparse.ArgumentParser(description="Stock Forecasting and Anomaly Detection Pipeline")
    parser.add_argument('--prophet', action='store_true', help='Run Prophet forecasting')
    parser.add_argument('--lstm', action='store_true', help='Run LSTM forecasting')
    parser.add_argument('--iforest', action='store_true', help='Run Isolation Forest anomaly detection')
    parser.add_argument('--all', action='store_true', help='Run all models')
    args = parser.parse_args()

    with open(os.path.join('config', 'config.yaml')) as f:
        config = yaml.safe_load(f)

    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(os.path.join(output_dir, 'pipeline.log'))
    logging.info("Pipeline started")

    df = load_data(config['paths']['data_path'])
    logging.info(f"Data loaded: {len(df)} records")

    if args.all or args.prophet:
        forecast_prophet = run_prophet(df, config)
        metrics_prophet = evaluate_forecast(forecast_prophet['actual'], forecast_prophet['predicted'])
        logging.info(f"Prophet metrics: {metrics_prophet}")

    if args.all or args.lstm:
        forecast_lstm = run_lstm(df, config)
        metrics_lstm = evaluate_forecast(forecast_lstm['actual'], forecast_lstm['predicted'])
        logging.info(f"LSTM metrics: {metrics_lstm}")

    if args.all or args.iforest:
        anomalies = run_isolation_forest(df, config)
        logging.info(f"Anomalies detected: {anomalies['anomaly'].sum()}")

    logging.info("Pipeline completed.")

if __name__ == "__main__":
    main()
