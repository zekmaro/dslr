import pandas as pd
import numpy as np
from src.models.train.visual_tools import plot_heat_map


def get_best_features(df: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
	"""
	Get the best features from the dataset.
	:param df: DataFrame
	:return: DataFrame with the best features
	"""
	df = df.drop(columns=drop_columns)
	corr = df.corr()

	print("Absolute correlation matrix:")
	abs_corr = corr.abs()
	print(abs_corr)
	plot_heat_map(abs_corr, "Absolute Correlation Matrix")

	best_features_list = []
	for i in range(len(abs_corr.columns)):
		for j in range(i + 1, len(abs_corr.columns)):
			val = abs_corr.iloc[i, j]
			if val > 0.6 or val < 0.3:
				feature1 = abs_corr.index[i]
				feature2 = abs_corr.columns[j]
				best_features_list.append((feature1, feature2, float(val)))

	print(f"Best features: {best_features_list}")
	return best_features_list
