"""
whole thing implemented with scikit
"""

# import sklearn
import pandas as pd

DATA_PATH = "./datasets/dataset_train.csv"

def main() -> None:
    """Load the trained model and evaluate its accuracy on the test set."""
    train_data = pd.read_csv(DATA_PATH)
    print(train_data.describe())


if __name__ == "__main__":
    main()
