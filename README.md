# Bangladesh Stock Forecasting and Anomaly Detection

This repository provides a complete pipeline for forecasting stock prices and detecting anomalies on Bangladesh stock market data.

## Structure

- data/: Sample dataset (CSV).
- models/: Model implementations for Prophet, LSTM, Isolation Forest.
- utils/: Preprocessing and evaluation utilities.
- config/: Configuration file (YAML).
- results/: Output plots and logs.
- main.py: CLI entry point.
- Dockerfile: Docker setup.
- requirements.txt: Dependencies.
- .gitignore: Ignored files.

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run pipeline:
   ```
   python main.py --all
   ```
3. Results in `results/`.

## Docker

```
docker build -t stock-forecast .
docker run --rm stock-forecast
```
