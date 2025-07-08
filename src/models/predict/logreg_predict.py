import json
import numpy as np
from src.utils.load_csv import load
from src.utils.header import TEST_DATASET_PATH


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_model_data(filename='shared_data/model.json'):
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
    model_data = load_model_data()
    mean, std, weights = model_data["mean"], model_data["std"], model_data["weights"]
    test_df = load(TEST_DATASET_PATH)
    print(test_df)
    training_features = ['Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Charms', 'Flying']
    cleaned_df = test_df.dropna(subset=training_features)
    student_data = cleaned_df[training_features].to_numpy()
    print(student_data)
    normalized_data = (student_data - mean) / std
    print(normalized_data.shape)

    predicted_houses = predict(normalized_data, weights)


if __name__ == "__main__":
    main()