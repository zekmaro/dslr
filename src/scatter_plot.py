import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def plot_features(data: pd.DataFrame, feature_1: pd.DataFrame, feature_2: pd.DataFrame):
	house_colors = {
		"Gryffindor": "red",
		"Slytherin": "green",
		"Ravenclaw": "blue",
		"Hufflepuff": "orange"
	}
	for house, color in house_colors.items():
		subset = data[data["Hogwarts House"] == house]
		plt.scatter(subset[feature_1], subset[feature_2], label=house, color=color, alpha=0.6, s=15)
	plt.xlabel(feature_1)
	plt.ylabel(feature_2)
	plt.title(f"{feature_1} vs {feature_2}")
	plt.legend()
	plt.grid(True)
	plt.show()


def find_similar_features(df: pd.DataFrame):
	"""Find similar features in a DataFrame based on mean and standard deviation."""
	num_columns = [col for col in df.columns if df[col].dtype in [int, float]]
	col_data = {
		col: {
			"mean": df[col].mean(),
			"std": df[col].std()
		} for col in num_columns
	}
	similar_features = []
	for col1, data1 in col_data.items():
		for col2, data2 in col_data.items():
			if col1 != col2 and abs(data1["mean"] - data2["mean"]) < 1 and abs(data1["std"] - data2["std"]) < 1:
				if not (col1, col2) in similar_features and not (col2, col1) in similar_features:
					similar_features.append((col1, col2))
					plot_features(df, col1, col2)
	return similar_features
