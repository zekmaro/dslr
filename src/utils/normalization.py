from src.utils.header import TRAINING_FEATURES
from typing import List
import pandas as pd
import numpy as np


def normalize_data(
    x_clean: pd.DataFrame,
    feature_names: List[str],
) -> np.ndarray:
    """
    Normalize the student data by calculating the mean and standard deviation.

    Args:
        x_clean (pd.DataFrame): The cleaned training data.
        feature_means (np.ndarray): The means of the features.
        feature_stds (np.ndarray): The standard deviations of the features.
    
    Returns:
        np.ndarray: The normalized student data.
    """
    feature_means = x_clean.mean(axis=0)
    feature_stds = x_clean.std(axis=0)
    student_data = x_clean[feature_names].to_numpy()
    normalized_data = (student_data - feature_means) / feature_stds


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
