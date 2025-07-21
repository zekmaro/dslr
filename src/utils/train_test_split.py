import pandas as pd
import numpy as np


def train_test_split(
    x: pd.DataFrame,
    y: np.ndarray,
    split: float = 0.7,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    assert 0 < split < 1 and len(x) == len(y)

    indices = np.arange(len(x))
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(len(x) * split)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    x_train, y_train = x.iloc[train_idx], y[train_idx]
    x_test, y_test = x.iloc[test_idx], y[test_idx]

    return x_train, y_train, x_test, y_test
