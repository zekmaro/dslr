from models.OneVsRestClassifier import OneVsRestClassifier
from models.LogisticRegression import LogisticRegression
from utils.train_test_split import train_test_split
from utils.normalization import clean_data
from utils.load_csv import load
from utils.header import (
    TRAIN_DATASET_PATH,
    DROP_COLS,
    TRAINING_FEATURES
)


def main() -> None:
    """Main function to train the logistic regression model for each house."""
    df = load(TRAIN_DATASET_PATH)

    x = df.drop(columns=DROP_COLS + ["Hogwarts House"])
    y = df["Hogwarts House"]

    x_train, y_train, _, _ = train_test_split(x, y.to_numpy())  # test set not used during training.

    x_clean, y_clean = clean_data(x_train, y_train, TRAINING_FEATURES)

    ovr = OneVsRestClassifier(LogisticRegression)
    ovr.fit(x_clean, y_clean.to_numpy(), TRAINING_FEATURES)

    ovr.save_model_to_file()


if __name__ == "__main__":
    main()
