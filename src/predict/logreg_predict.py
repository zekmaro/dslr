from src.utils.header import (
    TRAIN_DATASET_PATH,
    DROP_COLS,
    HOUSE_MAP,
    TRAINING_FEATURES
)
from src.models.LogisticRegression import LogisticRegression
from src.models.OneVsRestClassifier import OneVsRestClassifier
from src.utils.train_test_split import train_test_split
from src.utils.normalization import clean_data
from src.utils.load_csv import load
import numpy as np


def main() -> None:
    """Main function to load the weights and print them."""
    df = load(TRAIN_DATASET_PATH)
    x_train, y_train, x_test, y_test = train_test_split(df.drop(columns=DROP_COLS), df["Hogwarts House"].to_numpy())

    ovr = OneVsRestClassifier(LogisticRegression)
    ovr.load_model_from_file()

    x_clean, y_clean = clean_data(x_train, y_train, TRAINING_FEATURES)
    normalized_data = ovr.normalize_data(x_clean, TRAINING_FEATURES)

    predicted_houses = ovr.predict(normalized_data)
    prediction_vector = np.array([HOUSE_MAP[house] for house in predicted_houses])

    true_vector = y_clean.map(HOUSE_MAP).to_numpy()
    accuracy = np.mean(prediction_vector == true_vector)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()