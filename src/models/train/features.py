import pandas as pd
import numpy as np


def get_best_features(df: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
	"""
	Get the best features from the dataset.
	:param df: DataFrame
	:return: DataFrame with the best features
	"""
	df = df.drop(columns=drop_columns)

	corr = df.corr()
	print("Correlation matrix:")
	print(corr)

	print("Absolute correlation matrix:")
	abs_corr = corr.abs()
	print(abs_corr)
