import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import logging

def run_prophet(df, config):
    logging.info("Starting Prophet forecasting...")
    df_prophet = df.rename(columns={'Date': 'ds', 'Close': 'y'}).copy()
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    forecast_days = config['prophet']['forecast_days']
    df_train = df_prophet.iloc[:-forecast_days]
    df_test = df_prophet.iloc[-forecast_days:]
    model = Prophet(daily_seasonality=True)
    model.fit(df_train)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    forecast_df = forecast.set_index('ds')[['yhat']].merge(
        df_prophet.set_index('ds')[['y']], left_index=True, right_index=True, how='left'
    ).reset_index()
    forecast_eval = forecast_df[forecast_df['ds'] > df_train['ds'].max()].copy()
    forecast_eval.rename(columns={'y': 'actual', 'yhat': 'predicted'}, inplace=True)
    plt.figure(figsize=(10,6))
    plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    plt.axvline(x=df_train['ds'].max(), color='red', linestyle='--', label='Forecast Start')
    plt.xlabel('Date'); plt.ylabel('Price'); plt.title('Prophet Forecast')
    plt.legend()
    plt.tight_layout()
    plt.savefig(config['paths']['output_dir'] + '/prophet_forecast.png')
    plt.close()
    logging.info("Prophet forecasting completed. Plot saved.")
    return forecast_eval
