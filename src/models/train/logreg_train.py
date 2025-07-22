from src.utils.header import (
	TRAIN_DATASET_PATH,
    DROP_COLS,
    HOUSE_MAP,
    MODEL_DATA_PATH,
    TRAINING_FEATURES
)
from src.models.train.LogisticRegression import LogisticRegression
from src.models.train.visual_tools import plot_cost_history
from src.utils.train_test_split import train_test_split
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


def load_weights(
    weights: Dict[str, List[float]],
    mean: np.ndarray,
    std: np.ndarray,
    filename: str = MODEL_DATA_PATH
) -> None:
    """
    Load the weights into a JSON file.

    Args:
        weights (Dict[str, List[float]]): The list of weights for each house.
        mean (np.ndarray): Array of means for each feature of the training data.
        std (np.ndarray): Array of standard deviations for each feature of the training data.
        filename (str): The path to the JSON file where the weights will be saved.
    
    Returns:
        None
    """
    print(mean, std)
    model_data = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights
    }
    with open(filename, "w") as f:
        json.dump(model_data, f)


def main():
    """Main function to train the logistic regression model for each house."""
    df = load(TRAIN_DATASET_PATH)

    x = df.drop(columns=DROP_COLS + ["Hogwarts House"])
    y = df["Hogwarts House"]

    x_train, y_train, x_test, y_test = train_test_split(x, y.to_numpy())

    x_train_clean = x_train.dropna(subset=TRAINING_FEATURES)
    y_train_series = pd.Series(y_train, index=x_train.index)
    y_train_clean = y_train_series.loc[x_train_clean.index]

    mean, std, normalized_data = normalize_student_data(x_train_clean)

    house_indices = pd.Series(y_train_clean).map(HOUSE_MAP).to_numpy()
    targets = {
        "Gryffindor": (house_indices == 0).astype(int),
        "Slytherin": (house_indices == 1).astype(int),
        "Ravenclaw": (house_indices == 2).astype(int),
        "Hufflepuff": (house_indices == 3).astype(int),
    }

    output_weights = {}
    for house, target in targets.items():
        model = LogisticRegression(learning_rate=0.1, iterations=1000, track_cost=False)
        model.fit(normalized_data, target)
        if model.track_cost:
            plot_cost_history(model.cost_history, title=f"Training Cost Over Time for {house}")
        output_weights[house] = model.weights.tolist()

    load_weights(output_weights, mean, std)


if __name__ == "__main__":
    main()
