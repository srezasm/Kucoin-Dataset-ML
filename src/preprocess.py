import pandas as pd
import numpy as np
import joblib
import sys
from config import Config

from sklearn.model_selection import train_test_split


def is_ascending(op, cp):
    return 1 if op < cp else 0


def is_morning_or_evening_star(op, cp, lp, hp, prev_op, prev_cp, prev_lp, prev_hp):
    # bearish evening star
    if prev_cp > prev_op:
        # Check if the current candle is a small-bodied candle
        if (cp - op) < 0.25 * (hp - lp):
            if op > prev_hp:
                # Check if the current candle closes below the previous candle's midpoint
                if cp < (prev_op + prev_cp) / 2:
                    return 1
    # bullish morning star
    elif prev_cp < prev_op:
        # Check if the current candle is a small-bodied candle
        if (cp - op) < 0.25 * (hp - lp):
            if op < prev_lp:
                # Check if the current candle closes above the previous candle's midpoint
                if cp > (prev_op + prev_cp) / 2:
                    return 2

    return 0


# https://yourfinancebook.com/marubozu-candlestick-pattern/
def is_marubozu(op, cp, lp, hp):
    body_range = abs(cp - op)
    upper_shadow = hp - max(op, cp)
    lower_shadow = min(op, cp) - lp
    is_asc = is_ascending(op, cp)

    # Check if the candlestick has no upper or lower shadow
    if upper_shadow == 0 and lower_shadow == 0:
        # marubozu
        if is_asc:
            return 1
        else:
            return 2

    # Check if the candlestick's body covers at least 95% of the candlestick's range
    if body_range >= 0.95 * (hp - lp):
        # marubozu open
        if is_asc:
            if upper_shadow > lower_shadow:
                return 3
            else:
                return 4
        # marubozu close
        else:
            if upper_shadow > lower_shadow:
                return 5
            else:
                return 6

    return 0


# https://howtotradeblog.com/what-is-pin-bar-candlestick/
def is_pin_bar(op, cp, lp, hp):
    body_range = abs(cp - op)
    upper_shadow = hp - max(op, cp)
    lower_shadow = min(op, cp) - lp
    is_asc = is_ascending(op, cp)

    # the upper shadow is at least twice as long as the body
    if upper_shadow >= 2 * body_range:
        # little or no lower shadow
        if lower_shadow <= 0.1 * body_range:
            if is_asc:
                return 1
            else:
                return 2

    # the lower shadow is at least twice as long as the body
    if lower_shadow >= 2 * body_range:
        # Check if there is little or no upper shadow
        if upper_shadow <= 0.1 * body_range:
            if is_asc:
                return 3
            else:
                return 4

    return 0


# https://srading.com/doji-candlestick-patterns/
def detect_doji(op, cp, lp, hp):
    body_size = abs(cp - op)
    shadow_size = max(hp, op, cp) - min(lp, op, cp)

    # standard doji
    if body_size < 0.1 * shadow_size:
        return 1

    # long-legged doji
    if 0.1 * shadow_size <= body_size < 0.3 * shadow_size:
        return 2

    # dragonfly doji
    if cp == hp and body_size < 0.1 * shadow_size:
        return 3

    # gravestone doji
    if op == lp and body_size < 0.1 * shadow_size:
        return 4

    return 0


# TODO: PRZ
def create_features(csv_file):
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        if lines[0].startswith('https'):
            lines.pop(0)
            with open(csv_file, 'w') as f:
                f.writelines(lines)

    data = pd.read_csv(csv_file)

    data = data[['open', 'close', 'low', 'high']]
    data = data.reindex(index=data.index[::-1])
    data = np.array(data, 'f16')

    X = []
    y = []
    for i in range(len(data) - 1):
        X.append(
            [
                is_morning_or_evening_star(*data[i], *data[i-1]),
                is_marubozu(*data[i]),
                is_pin_bar(*data[i]),
                detect_doji(*data[i]),
                *data[i]
            ]
        )

        y.append(is_ascending(*data[i, [0, 1]]))

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
