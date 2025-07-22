from src.utils.header import TRAINING_FEATURES
from typing import List
import pandas as pd
import numpy as np


def normalize_student_data(x_clean: pd.DataFrame) -> np.ndarray:
    """
    Normalize the student data by calculating the mean and standard deviation.

    Args:
        x_clean (pd.DataFrame): The cleaned training data.
    
    Returns:
        np.ndarray: The normalized student data.
    """
    student_data = x_clean[TRAINING_FEATURES].to_numpy()
    mean = student_data.mean(axis=0)
    std = student_data.std(axis=0)
    normalized_data = (student_data - mean) / std
    return (mean, std, normalized_data)


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
