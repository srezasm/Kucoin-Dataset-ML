import pandas as pd
import numpy as np
import joblib
import sys
from config import Config

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def create_features(csv_file):
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        if lines[0].startswith('https'):
            lines.pop(0)
            with open(csv_file, 'w') as f:
                f.writelines(lines)

    data = pd.read_csv(csv_file)

    cols = ['close', 'high', 'open', 'low']
    data = data[cols]
    data = data.reindex(index=data.index[::-1])
    data = np.array(data)

    X = []
    y = []
    for i in range(len(data) - 1):
        X.append(data[i])
        if data[i+1][0] > data[i][0]:
            y.append(1)  # ascending
        else:
            y.append(0)  # descending

    X = np.array(X)
    y = np.array(y)

    return train_test_split(
        X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    config = Config(csv_file)

    joblib.dump(create_features(csv_file), config.features_path)
