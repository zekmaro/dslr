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
    _, _, x_test, y_test = train_test_split(
        df.drop(columns=DROP_COLS),
        df["Hogwarts House"].to_numpy()
    )
    x_clean, y_clean = clean_data(x_test, y_test, TRAINING_FEATURES)

    ovr = OneVsRestClassifier(LogisticRegression)
    ovr.load_model_from_file()

    normalized_x = ovr.normalize_data(x_clean, TRAINING_FEATURES)
    predictions = ovr.predict(normalized_x)

    predicted_labels = np.array([HOUSE_MAP[label] for label in predictions])
    true_labels = y_clean.map(HOUSE_MAP).to_numpy()

    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()