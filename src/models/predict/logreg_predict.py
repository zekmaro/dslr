import json
from src.utils.load_csv import load
from src.utils.header import TEST_DATASET_PATH

def load_model_data(filename='shared_data/weights.json'):
    """
    Load the weights from a JSON file.
    :param filename: The path to the JSON file containing the weights
    :return: The weights as a dictionary
    """
    with open(filename, "r") as f:
        model_data = json.load(f)
    return model_data


def main():
    """
    Main function to load the weights and print them.
    :return: None
    """
    model_data = load_model_data()
    print("Weights loaded successfully:")
    print(model_data)
    mean, std, weights = model_data["mean"], model_data["std"], model_data["weights"]
    test_df = load(TEST_DATASET_PATH)

