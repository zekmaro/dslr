from typing import List
import pandas as pd
import numpy as np


def clean_data(
    x: pd.DataFrame,
    y: np.ndarray,
    features: List[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean the data by dropping rows with NaN values in specified features.
    
    Args:
		x (pd.DataFrame): The feature DataFrame.
		y (np.ndarray): The target vector.
		features (List[str]): The list of features to check for NaN values.

    Returns:
		tuple[pd.DataFrame, pd.Series]: Cleaned feature DataFrame and target Series.
    """
    x_clean = x.dropna(subset=features)
    y_series = pd.Series(y, index=x.index)
    y_clean = y_series.loc[x_clean.index]
    return x_clean, y_clean
