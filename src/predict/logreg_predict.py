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


def main() -> None:
    """Load the trained model and evaluate its accuracy on the test set."""
    df = load(TRAIN_DATASET_PATH)

    _, _, x_test, y_test = train_test_split(
        df.drop(columns=DROP_COLS),
        df["Hogwarts House"].to_numpy()
    ) # train set not used during training.

    x_clean, y_clean = clean_data(x_test, y_test, TRAINING_FEATURES)

    ovr = OneVsRestClassifier(LogisticRegression)
    ovr.load_model_from_file()

    score = ovr.evaluate(x_clean, y_clean, TRAINING_FEATURES)
    print(f"Model accuracy: {score * 100:.2f}%")


if __name__ == "__main__":
    main()