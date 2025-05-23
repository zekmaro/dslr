from src.utils.load_csv import load
from src.utils.header import DATASET_PATH, DROP_COLS, HOUSE_MAP
from src.models.train.features import get_best_features, rm_redundant_features
from src.models.train.training import gradient_descent


def main():
	df = load(DATASET_PATH)
	best_features = get_best_features(df, DROP_COLS)
	print()
	features_to_remove = rm_redundant_features(df, best_features)
	training_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in DROP_COLS and col not in features_to_remove]
	print(f"Numerous features: {training_features}")
	needed_columns = training_features + ['Hogwarts House']
	cleaned_df = df.dropna(subset=needed_columns)
	student_data = cleaned_df[training_features].dropna().to_numpy()
	print(f"Student data before normalization: {student_data}")
	print(f"Student data shape: {student_data.shape}")
	mean = student_data.mean(axis=0)
	std = student_data.std(axis=0)
	normalized_data = (student_data - mean) / std
	print(f"Normalized data: {normalized_data}")
	houses_vector = cleaned_df['Hogwarts House'].map(HOUSE_MAP).to_numpy()
	print(f"House vector: {houses_vector}")
	print(f"House vector shape: {houses_vector.shape}")

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

	output_weights = {"Gryffindor": None, "Slytherin": None, "Ravenclaw": None, "Hufflepuff": None}
	for house, target in targets.items():
		print(f"Training for {house}...")
		weights = gradient_descent(normalized_data, target, alpha=0.1, epochs=1000)
		output_weights[house] = weights
		print(f"Weights for {house}: {weights}")

	print(f"Weights: {output_weights}")

if __name__ == "__main__":
	main()
