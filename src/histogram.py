import pandas as pd
import matplotlib.pyplot as plt


def plot_histogram(df: pd.DataFrame, column: str, bins: int = 10) -> None:
	"""Plot a histogram of a specified column in a DataFrame.

	Args:
		df (pd.DataFrame): The DataFrame containing the data.
		column (str): The column name to plot.
		bins (int, optional): The number of bins for the histogram. Defaults to 10.
	"""
	if column not in df.columns:
		raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

	plt.figure(figsize=(10, 6))
	plt.hist(df[column], bins=bins, edgecolor='black')
	plt.title(f"Histogram of {column}")
	plt.xlabel(column)
	plt.ylabel("Frequency")
	plt.grid(axis='y', alpha=0.75)
	plt.show()
