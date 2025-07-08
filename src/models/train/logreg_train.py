from src.utils.load_csv import load
from src.utils.header import TRAIN_DATASET_PATH, DROP_COLS, HOUSE_MAP, TEST_DATASET_PATH
from src.models.train.features import get_best_features, rm_redundant_features
from src.models.train.training import gradient_descent
import numpy as np
import json


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


def get_training_features(df):
    best_features = get_best_features(df, DROP_COLS)
    print()
    features_to_remove = rm_redundant_features(df, best_features)
    training_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in DROP_COLS and col not in features_to_remove]
    print(f"Traning features: {training_features}")
    print()
    return training_features


def normalize_student_data(cleaned_df, training_features):
    student_data = cleaned_df[training_features].to_numpy()
    mean = student_data.mean(axis=0)
    std = student_data.std(axis=0)
    normalized_data = (student_data - mean) / std
    return normalized_data


def load_weights(weights, filename='shared_data/weights.json'):
    with open(filename, "w") as f:
        json.dump(weights, f)


def main():
    df = load(TRAIN_DATASET_PATH)

    training_features = get_training_features(df)
    needed_columns = training_features + ['Hogwarts House']
    cleaned_df = df.dropna(subset=needed_columns)
    normalized_data = normalize_student_data(cleaned_df, training_features)
    houses_vector = cleaned_df['Hogwarts House'].map(HOUSE_MAP).to_numpy()

    target_gryffindor = (houses_vector == 0).astype(int)
    target_slytherin = (houses_vector == 1).astype(int)
    target_ravenclaw = (houses_vector == 2).astype(int)
    target_hufflepuff = (houses_vector == 3).astype(int)

    targets = {
        "Gryffindor": target_gryffindor, 
        "Slytherin": target_slytherin,
        "Ravenclaw": target_ravenclaw,
        "Hufflepuff": target_hufflepuff
    }

    output_weights = {
        "Gryffindor": None,
        "Slytherin": None,
        "Ravenclaw": None,
        "Hufflepuff": None
    }

    for house, target in targets.items():
        print(f"Training for {house}...")
        weights = gradient_descent(normalized_data, target, alpha=0.1, epochs=1000)
        output_weights[house] = weights.tolist()
        print(f"Weights for {house}: {weights}")
        print()

    load_weights(output_weights)

if __name__ == "__main__":
    main()
