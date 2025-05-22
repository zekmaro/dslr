from src.utils.load_csv import load
from src.utils.header import DATASET_PATH, DROP_COLS
from src.models.train.features import get_best_features, rm_redundant_features
from src.models.train.features import plot_feature_distribution


def main():
	df = load(DATASET_PATH)
	best_features = get_best_features(df, DROP_COLS)
	print()
	redundant = rm_redundant_features(df, best_features)

if __name__ == "__main__":
	main()
