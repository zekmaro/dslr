from src.utils.load_csv import load
from src.utils.header import TRAIN_DATASET_PATH, DROP_COLS, HOUSE_MAP, MODEL_DATA_PATH, TRAINING_FEATURES, COURSES
from src.models.train.training import gradient_descent
import numpy as np
import json
from src.utils.train_test_split import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from src.models.train.LogisticRegression import LogisticRegression
from src.models.train.visual_tools import plot_cost_history

def predict_house(student_data, weights):
    """
    Predict the house of a student based on their data and the weights.
    :param student_data: The student's data
    :param weights: The weights for the model
    :return: The predicted house
    """
    probabilities = {}
    for house, weight in weights.items():
        probabilities[house] = 1 / (1 + np.exp(-np.dot(student_data, weight)))
    return max(probabilities, key=probabilities.get)


def predict_test_data(test_df, training_features, output_weights):
    """
    Predict the house for the test data.
    :param test_df: The test DataFrame
    :param training_features: The features used for training
    :param output_weights: The weights for the model
    :return: The predicted houses for the test data
    """
    test_data = test_df[training_features].dropna().to_numpy()
    predictions = []
    for student in test_data:
        predicted_house = predict_house(student, output_weights)
        predictions.append(predicted_house)
    return predictions


def normalize_student_data(cleaned_df, training_features):
    student_data = cleaned_df[training_features].to_numpy()
    mean = student_data.mean(axis=0)
    std = student_data.std(axis=0)
    normalized_data = (student_data - mean) / std
    return (mean, std, normalized_data)


def load_weights(weights, mean, std, filename=MODEL_DATA_PATH):
    print(mean, std)
    model_data = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights
    }
    with open(filename, "w") as f:
        json.dump(model_data, f)


def main():
    df = load(TRAIN_DATASET_PATH)

    # 1. Split into training and testing
    x = df.drop(columns=DROP_COLS + ["Hogwarts House"])
    y = df["Hogwarts House"]
    x_train, y_train, x_test, y_test = train_test_split(x, y.to_numpy())

    # 3. Drop NaNs in training data
    x_train_clean = x_train.dropna(subset=TRAINING_FEATURES)
    y_train_series = pd.Series(y_train, index=x_train.index)
    y_train_clean = y_train_series.loc[x_train_clean.index]

    # 4. Normalize training features
    student_data = x_train_clean[TRAINING_FEATURES].to_numpy()
    mean = student_data.mean(axis=0)
    std = student_data.std(axis=0)
    normalized_data = (student_data - mean) / std

    # 5. One-vs-all target vectors
    house_indices = pd.Series(y_train_clean).map(HOUSE_MAP).to_numpy()
    targets = {
        "Gryffindor": (house_indices == 0).astype(int),
        "Slytherin": (house_indices == 1).astype(int),
        "Ravenclaw": (house_indices == 2).astype(int),
        "Hufflepuff": (house_indices == 3).astype(int),
    }

    # 6. Train for each house
    output_weights = {}
    for house, target in targets.items():
        # print(f"Training for {house}...")
        model = LogisticRegression(learning_rate=0.1, iterations=1000, track_cost=True)
        model.fit(normalized_data, target)
        weights = model.weights
        if model.track_cost:
            plot_cost_history(model.cost_history, title=house)
        output_weights[house] = weights.tolist()
        # print(f"Weights for {house}: {weights}\n")

    # 7. Save the model
    load_weights(output_weights, mean, std)


if __name__ == "__main__":
    main()
