from src.utils.header import (
	TRAIN_DATASET_PATH,
    DROP_COLS,
    TRAINING_FEATURES
)
from src.models.LogisticRegression import LogisticRegression
from src.models.OneVsRestClassifier import OneVsRestClassifier
from src.utils.train_test_split import train_test_split
from src.train.train_utils import normalize_student_data, clean_data
from src.utils.load_csv import load


def main():
    """Main function to train the logistic regression model for each house."""
    df = load(TRAIN_DATASET_PATH)

    x = df.drop(columns=DROP_COLS + ["Hogwarts House"])
    y = df["Hogwarts House"]

    x_train, y_train, x_test, y_test = train_test_split(x, y.to_numpy())
    x_clean, y_clean = clean_data(x_train, y_train, TRAINING_FEATURES)

    mean, std, normalized_data = normalize_student_data(x_clean)

    ovr = OneVsRestClassifier(LogisticRegression)
    ovr.fit(normalized_data, y_clean.to_numpy())
    ovr.safe_model_to_file(mean, std)


if __name__ == "__main__":
    main()
