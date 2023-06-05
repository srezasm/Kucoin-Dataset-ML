import pandas as pd
import numpy as np
import joblib
import sys
from config import Config

from sklearn.model_selection import train_test_split


def is_bullish_morning_star(open, close, low, high, prev_open, prev_close, prev_low):
    # Check if the previous candle is a bearish candle
    if prev_close < prev_open:
        # Check if the current candle is a small-bodied candle
        if (close - open) < 0.25 * (high - low):
            # Check if the current candle opens below the previous candle's low
            if open < prev_low:
                # Check if the current candle closes above the previous candle's midpoint
                if close > (prev_open + prev_close) / 2:
                    # The pattern is a bullish morning star
                    return 1

    # The pattern is not a bullish morning star
    return 0

def is_bearish_evening_star(open, close, low, high, prev_open, prev_close, prev_high):
    # Check if the previous candle is a bullish candle
    if prev_close > prev_open:
        # Check if the current candle is a small-bodied candle
        if (close - open) < 0.25 * (high - low):
            # Check if the current candle opens above the previous candle's high
            if open > prev_high:
                # Check if the current candle closes below the previous candle's midpoint
                if close < (prev_open + prev_close) / 2:
                    # The pattern is an evening bearish star
                    return 1

    # The pattern is not an evening bearish star
    return 0

# TODO: Candle stick patterns
# TODO: Marubozu candle stick
# TODO: Pin Bar candle stick
# TODO: Doji candle stick
# TODO: PRZ
def create_features(csv_file):
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        if lines[0].startswith('https'):
            lines.pop(0)
            with open(csv_file, 'w') as f:
                f.writelines(lines)

    data = pd.read_csv(csv_file)

    data = data[['open', 'high', 'low', 'close']]
    data = data.reindex(index=data.index[::-1])
    data = np.array(data, 'f16')

    X = []
    y = []
    for i in range(len(data) - 1):
        is_bms = is_bullish_morning_star(*data[i, [0, 3, 2, 1]],
                                         *data[i-1, [0, 3, 2]])
        is_bes = is_bearish_evening_star(*data[i, [0, 3, 2, 1]],
                                         *data[i-1, [0, 3, 1]])
        X.append(np.append(data[i], [is_bms, is_bes]))
        if is_bms == 1:
            print('bms', X[-1], y[-1])
        if is_bes == 1:
            print('bes' ,X[-1], y[-1])
        
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
