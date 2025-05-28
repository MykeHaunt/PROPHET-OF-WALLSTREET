import numpy as np
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), :])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)
