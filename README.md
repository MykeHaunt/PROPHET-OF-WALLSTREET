![IMG_0129](https://github.com/user-attachments/assets/56af6404-73f6-49ca-95e5-2f7b0068c9e8)
# PROPHET-OF-WALLSTREET

## Overview

**PROPHET-OF-WALLSTREET** is a Python-based stock market analysis toolkit designed to predict stock prices and detect anomalies in financial time-series data. It integrates traditional statistical forecasting models (such as Facebook’s Prophet), deep learning architectures (like LSTM neural networks), and anomaly detection algorithms (such as Isolation Forest) into a cohesive pipeline. The goal is to empower investors, analysts, and researchers with a transparent, data-driven toolset for financial forecasting and risk assessment.

---

## Core Features

- **Time-Series Forecasting**  
  Utilizes Facebook’s **Prophet** model to forecast future stock prices based on seasonal trends. Prophet handles missing data, outliers, and changepoints effectively.

- **Deep Learning via LSTM**  
  Applies **Long Short-Term Memory (LSTM)** neural networks to learn from historical stock price sequences and predict future trends.

- **Anomaly Detection**  
  Employs the **Isolation Forest** algorithm to flag data points that deviate significantly from the historical trend, helping detect potential manipulations or unusual market events.

- **Data Preprocessing**  
  Offers utilities for reading stock data, scaling numerical features, and generating training sequences for LSTM-based models.

- **Visualization**  
  Provides functions to visualize stock price trends and mark anomalies directly on the price chart for easier analysis.

---

## Technical Stack

| Category            | Tools/Frameworks                                |
|---------------------|--------------------------------------------------|
| Language            | Python 3.x                                       |
| Forecasting         | Facebook Prophet                                |
| Deep Learning       | LSTM (TensorFlow/Keras or PyTorch recommended)  |
| ML/Anomaly Detection| Scikit-learn (Isolation Forest)                 |
| Data Handling       | Pandas, NumPy                                   |
| Visualization       | Matplotlib                                      |

---

## Code Structure

### `preprocessing.py`

Contains functions for:
- **`load_stock_data(csv_path)`**: Reads CSV file and sorts by date.
- **`scale_data(df, feature_cols)`**: Normalizes selected features using MinMaxScaler.
- **`create_lstm_sequences(data, sequence_length)`**: Generates input-output pairs for LSTM training from normalized sequences.

### `isolation_forest.py`

Provides anomaly detection functionality:
- **`detect_anomalies(df, contamination=0.01)`**: Applies Isolation Forest to flag anomalies.
- **`plot_anomalies(df)`**: Visualizes anomalies on a line chart of closing prices.

### `sample_stock_data.csv`

A historical dataset containing:
- `date`, `open`, `high`, `low`, `close`, `volume`  
Spans one year of daily stock prices (approx. 261 rows).

---

## Algorithms Used

### Prophet

An additive time series forecasting model that handles daily, weekly, and yearly seasonality. Designed for interpretability and ease of use by business users.

### LSTM (Long Short-Term Memory)

A specialized form of recurrent neural network (RNN) capable of learning long-term dependencies. Ideal for sequential prediction tasks like stock price modeling.

### Isolation Forest

An unsupervised learning algorithm for anomaly detection. Efficiently isolates outliers using an ensemble of randomly partitioned trees.

---

## Sample Usage

### Load and Preprocess Data

```python
from preprocessing import load_stock_data, scale_data, create_lstm_sequences

df = load_stock_data("sample_stock_data.csv")
scaled_values, scaler = scale_data(df, feature_cols=["close"])
X, y = create_lstm_sequences(scaled_values, sequence_length=50)
```

### Apply Anomaly Detection

```python
from isolation_forest import detect_anomalies, plot_anomalies

df_with_scores = detect_anomalies(df, contamination=0.01)
plot_anomalies(df_with_scores)
```

### Build LSTM Model (example outline)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, input_shape=(50, 1)),
    Dense(1)
])

model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=10, batch_size=32)
```

---

## Visual Output

The tool produces:
- A time series plot of closing stock prices.
- Anomaly markers (in red) overlaid on the price trend.
- LSTM and Prophet forecasts (can be visualized separately as extensions).

---

## Installation & Setup

### Requirements

- Python 3.8+
- Recommended packages:

```bash
pip install pandas numpy scikit-learn matplotlib tensorflow prophet
```

### Running the Pipeline

1. Replace `sample_stock_data.csv` with your own stock data.
2. Call the preprocessing and modeling functions.
3. Visualize anomalies or predictions using provided functions or Jupyter notebooks.

---

## Potential Applications

- **Retail Investors**: To identify abnormal price behavior and inform buy/sell decisions.
- **Quant Researchers**: As a base framework for algorithmic strategy development.
- **Financial Advisors**: To generate automated forecasts and anomaly reports.
- **Risk Analysts**: For detecting early warning signals of market manipulation or volatility.

---

## Limitations & Extensions

- **Not Financial Advice**: This tool is for research and educational purposes.
- **No Live Feed Integration**: Requires CSV-based historical data input.
- **Basic LSTM Model**: Can be extended with additional features (e.g., technical indicators, macro data).
- **No License Specified**: Currently lacks an open-source license.

---

## License

GNU GENERAL PUBLIC LICENSE
                       Version 2, June 1991

 Copyright (C) 1989, 1991 Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

---

## Authors & Credits

Originally developed by **Myke Haunt**  
GitHub: [MykeHaunt/PROPHET-OF-WALLSTREET](https://github.com/MykeHaunt/PROPHET-OF-WALLSTREET)