from src.utils.header import (
    TRAIN_DATASET_PATH,
    DROP_COLS,
    MODEL_DATA_PATH,
    HOUSE_MAP,
    TRAINING_FEATURES
)
from src.models.LogisticRegression import LogisticRegression
from src.utils.train_test_split import train_test_split
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


def main():
    """
    Main function to load the weights and print them.
    :return: None
    """
    df = load(TRAIN_DATASET_PATH)
    x_train, y_train, x_test, y_test = train_test_split(df.drop(columns=DROP_COLS), df["Hogwarts House"].to_numpy())

    model = load_model_data()
    mean, std, weights = np.array(model["mean"]), np.array(model["std"]), model["weights"]

    x_test_clean = x_test.dropna(subset=TRAINING_FEATURES)
    y_test_series = pd.Series(y_test, index=x_test.index)
    y_test_clean = y_test_series.loc[x_test_clean.index]

    student_data = x_test_clean[TRAINING_FEATURES].to_numpy()
    normalized_data = (student_data - mean) / std

    predicted_houses = predict(normalized_data, weights)
    prediction_vector = np.array([HOUSE_MAP[house] for house in predicted_houses])

    true_vector = y_test_clean.map(HOUSE_MAP).to_numpy()
    accuracy = np.mean(prediction_vector == true_vector)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()