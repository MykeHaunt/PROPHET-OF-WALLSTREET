# Detailed Technical Description

## Overview  
The **Bangladesh Stock Market Forecasting & Anomaly Detection** repository is a modular proof‑of‑concept pipeline demonstrating advanced machine learning techniques applied to time series data. It is engineered in Python, leveraging industry‑standard libraries: TensorFlow/Keras (LSTM), Facebook Prophet, and scikit‑learn (Isolation Forest). The codebase is organized to facilitate reproducibility, extensibility, and compliance.

## 1. Data Simulation & Ingestion  
- **Synthetic Data Generation:** We simulate ~1 year of daily trading data for a hypothetical Bangladesh equity or index. Price series combine a deterministic trend component, seasonal cycles, volatility clustering, and randomly timed shock events. Volume data follow a log‑normal distribution calibrated to typical DSE volumes. A fixed random seed ensures reproducibility.  
- **Data Schema:** Each record contains `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`. Data are stored in CSV format under `data/sample_stock_data.csv`.  
- **Loading & Preprocessing:** The utility function `load_data()` reads CSV into a pandas DataFrame. Further preprocessing includes sorting by date, handling missing values (forward‑fill or zero imputation), and feature scaling (MinMaxScaler for LSTM).

## 2. Prophet Forecasting Module  
- **Input Preparation:**  
  1. Rename DataFrame columns: `Date→ds`, `Close→y`.  
  2. Convert `ds` to pandas datetime type.  
  3. Split into training and forecast periods based on `forecast_days` from `config.yaml`.  
- **Model Configuration:**  
  - `Prophet(daily_seasonality=True)`: Enables fine‑grained daily cycles.  
- **Training & Forecasting:**  
  1. Fit the model on historical data (`df_train`).  
  2. Generate a future DataFrame extending `n` days beyond training end.  
  3. Call `model.predict()` to obtain `yhat`, trend, seasonal, and residual components.  
- **Output:**  
  - A pandas DataFrame of date, `actual` (for test period), and `predicted`.  
  - Visualization: Matplotlib line chart of actual vs. forecast, saved under `results/prophet_forecast.png`.

## 3. LSTM Forecasting Module  
- **Sequence Creation:**  
  1. Scale `Close` values to [0,1] via `MinMaxScaler`.  
  2. Use a sliding window of length `window_size` (from config) to form input sequences (`X`) and labels (`y`).  
  3. Format `X` as 3D array `[samples, timesteps, features]`.  
- **Train/Test Split:** Perform an 80/20 split (or as configured) at the sequence level.  
- **Model Architecture (Keras Sequential):**  
```python
Sequential([
  LSTM(units=config['lstm']['units'], input_shape=(window_size, 1)),
  Dense(1)
])
```  
- **Training & Prediction:**  
  - Loss: Mean Squared Error.  
  - Optimizer: Adam with default learning rate.  
- **Output:**  
  - A DataFrame with `ds`, `actual`, and `predicted`.  
  - Plot of actual vs. predicted prices saved as `results/lstm_forecast.png`.

## 4. Isolation Forest Anomaly Detection Module  
- **Feature Selection:** Use `Close` price and `Volume` as features.  
- **Model Initialization:**  
```python
IsolationForest(
  n_estimators=config['isolation']['n_estimators'],
  contamination=config['isolation']['contamination'],
  random_state=42
)
```  
- **Training & Prediction:** Fit on features and predict anomaly flags.  
- **Output:** DataFrame of `Date`, `Close`, `Volume`, and `anomaly`; anomaly chart saved as `results/isolation_forest.png`.

## 5. Utility Modules  
- **preprocessing.py:**  
  - `load_data()`: CSV loader.  
  - `create_sequences(data, window_size)`: Sliding window generator for LSTM.  
- **evaluation.py:**  
  - `evaluate_forecast(actual, predicted)`: Returns MAE, RMSE, MAPE, and R².

## 6. Configuration Management  
YAML file (`config/config.yaml`) captures all paths and parameters, decoupling code and configuration for easy experimentation and auditability.

## 7. Orchestration & CLI  
- **main.py:** Parses flags `--prophet`, `--lstm`, `--iforest`, `--all`.  
- **Workflow:** Loads config, sets up logging, loads data, executes selected modules, logs metrics to `results/pipeline.log`.

## 8. Containerization & Deployment  
- **Dockerfile:**  
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "main.py", "--all"]
```
- **Usage:**  
```bash
docker build -t bd-stock-forecast .
docker run --rm -v $(pwd):/app bd-stock-forecast
```

## 9. Reproducibility & Best Practices  
- Fixed random seeds, Git version control, modular codebase, audit logs.

## 10. Extension Points  
Guidelines to integrate new models, real data, hyperparameter tuning, dashboards, and CI/CD pipelines.

*End of Technical Description.*
