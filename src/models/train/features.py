import pandas as pd
import numpy as np
from src.models.train.visual_tools import plot_heat_map, plot_graph


def rm_redundant_features(best_features: list[tuple[pd.DataFrame, pd.DataFrame, float]]) -> list[pd.DataFrame]:
	"""
	Remove redundant features from the list of best features.
	:param best_features: List of best features
	:return: List of best features without redundancy
	"""
	redundant_tuples = [(feature1, feature2, correlation) for feature1, feature2, correlation in best_features if correlation > 0.6]
	print(f"Redundant features: {redundant_tuples}")


def get_best_features(df: pd.DataFrame, drop_columns: list[str]) -> list[tuple[pd.DataFrame, pd.DataFrame, float]]:
	"""
	Get the best features from the dataset.
	:param df: DataFrame
	:return: DataFrame with the best features
	"""
	df = df.drop(columns=drop_columns)
	corr = df.corr()

	print("Absolute correlation matrix:")
	abs_corr = corr.abs()

	filtered_corr = abs_corr.copy()
	filtered_corr[(filtered_corr >= 0.3) & (filtered_corr <= 0.6)] = 0
	plot_graph(filtered_corr, "Absolute Correlation Graph")

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
