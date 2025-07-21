import json
import numpy as np
import pandas as pd
from src.utils.load_csv import load
from src.utils.header import TRAIN_DATASET_PATH, DROP_COLS, MODEL_DATA_PATH, HOUSE_MAP, TRAINING_FEATURES
from src.utils.train_test_split import train_test_split


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
    print(f"Training data shape: {x_train.shape}, Test data shape: {x_test.shape}, Training labels shape: {y_train.shape}, Test labels shape: {y_test.shape}")

    model = load_model_data()
    mean = np.array(model["mean"])
    std = np.array(model["std"])
    weights = model["weights"]

    # training_features = ['Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Charms', 'Flying']
    training_features = TRAINING_FEATURES
    x_test_clean = x_test.dropna(subset=training_features)

    y_test_series = pd.Series(y_test, index=x_test.index)
    y_test_clean = y_test_series.loc[x_test_clean.index]

    print(y_test_clean.value_counts())


    student_data = x_test_clean[training_features].to_numpy()
    normalized_data = (student_data - mean) / std

    predicted_houses = predict(normalized_data, weights)
    print(pd.Series(predicted_houses).value_counts())
    prediction_vector = np.array([HOUSE_MAP[house] for house in predicted_houses])

    true_vector = y_test_clean.map(HOUSE_MAP).to_numpy()
    accuracy = np.mean(prediction_vector == true_vector)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()