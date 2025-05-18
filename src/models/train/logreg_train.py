from src.utils.load_csv import load
from src.utils.header import DATASET_PATH, DROP_COLS
from src.models.train.features import get_best_features


def main():
	df = load(DATASET_PATH)
	get_best_features(df, DROP_COLS)
	


if __name__ == "__main__":
	main()
