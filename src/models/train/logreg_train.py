from src.utils.load_csv import load
from src.utils.header import DATASET_PATH, DROP_COLS, HOUSE_MAP
from src.models.train.features import get_best_features, rm_redundant_features


def main():
	df = load(DATASET_PATH)
	best_features = get_best_features(df, DROP_COLS)
	print()
	features_to_remove = rm_redundant_features(df, best_features)
	training_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in DROP_COLS and col not in features_to_remove]
	print(f"Numerous features: {training_features}")
	student_data = df[training_features].dropna().to_numpy()
	mean = student_data.mean(axis=0)
	std = student_data.std(axis=0)
	normalized_data = (student_data - mean) / std
	print(f"Normalized data: {normalized_data}")
	houses_vector = df['Hogwarts House'].map(HOUSE_MAP).to_numpy()
	print(f"House vector: {houses_vector}")


if __name__ == "__main__":
	main()
