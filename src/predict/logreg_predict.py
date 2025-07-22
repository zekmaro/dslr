from src.utils.header import (
    TRAIN_DATASET_PATH,
    DROP_COLS,
    MODEL_DATA_PATH,
    HOUSE_MAP,
    TRAINING_FEATURES
)
from src.models.LogisticRegression import LogisticRegression
from src.models.OneVsRestClassifier import OneVsRestClassifier
from src.utils.train_test_split import train_test_split
from src.utils.normalization import normalize_student_data, clean_data
from src.utils.load_csv import load
import pandas as pd
import numpy as np
import json


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_model_data(filename=MODEL_DATA_PATH):
    """
    Load the weights from a JSON file.
    :param filename: The path to the JSON file containing the weights
    :return: The weights as a dictionary
    """
    with open(filename, "r") as f:
        model_data = json.load(f)
    return model_data


def predict(normalized_data, house_weights):
    predicted_houses = []
    for row in normalized_data:
        houses_probabilities = {}
        for house, weights in house_weights.items():
            houses_probabilities[house] = sigmoid(np.dot(row, weights))
        best_house = max(houses_probabilities, key=houses_probabilities.get)  #type: ignore
        predicted_houses.append(best_house)
    return predicted_houses


def main() -> None:
    """Main function to load the weights and print them."""
    df = load(TRAIN_DATASET_PATH)
    x_train, y_train, x_test, y_test = train_test_split(df.drop(columns=DROP_COLS), df["Hogwarts House"].to_numpy())

    model = load_model_data()
    feature_means, feature_std, weights = np.array(model["mean"]), np.array(model["std"]), model["weights"]

    x_clean, y_clean = clean_data(x_train, y_train, TRAINING_FEATURES)
    normalized_data = normalize_student_data(x_clean, feature_means, feature_std)

    predicted_houses = predict(normalized_data, weights)
    prediction_vector = np.array([HOUSE_MAP[house] for house in predicted_houses])

    true_vector = y_clean.map(HOUSE_MAP).to_numpy()
    accuracy = np.mean(prediction_vector == true_vector)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()