from src.utils.header import (
	TRAIN_DATASET_PATH,
    DROP_COLS,
    HOUSE_MAP,
    MODEL_DATA_PATH,
    TRAINING_FEATURES
)
from src.models.LogisticRegression import LogisticRegression
from src.models.OneVsRestClassifier import OneVsRestClassifier
from src.utils.train_test_split import train_test_split
from src.train.visual_tools import plot_cost_history
from src.utils.load_csv import load
from typing import Dict, List
import pandas as pd
import numpy as np
import json


def normalize_student_data(x_train_clean: pd.DataFrame) -> np.ndarray:
    """
    Normalize the student data by calculating the mean and standard deviation.

    Args:
        x_train_clean (pd.DataFrame): The cleaned training data.
    
    Returns:
        np.ndarray: The normalized student data.
    """
    student_data = x_train_clean[TRAINING_FEATURES].to_numpy()
    mean = student_data.mean(axis=0)
    std = student_data.std(axis=0)
    normalized_data = (student_data - mean) / std
    return (mean, std, normalized_data)


def clean_data(x, y, features):
    x_clean = x.dropna(subset=features)
    y_series = pd.Series(y, index=x.index)
    y_clean = y_series.loc[x_clean.index]
    return x_clean, y_clean


def main():
    """Main function to train the logistic regression model for each house."""
    df = load(TRAIN_DATASET_PATH)

    x = df.drop(columns=DROP_COLS + ["Hogwarts House"])
    y = df["Hogwarts House"]

    x_train, y_train, x_test, y_test = train_test_split(x, y.to_numpy())
    x_clean, y_clean = clean_data(x_train, y_train, TRAINING_FEATURES)

    mean, std, normalized_data = normalize_student_data(x_clean)

    ovr = OneVsRestClassifier(LogisticRegression)
    ovr.fit(normalized_data, y_clean.to_numpy())
    ovr.safe_model_to_file(mean, std)


if __name__ == "__main__":
    main()
