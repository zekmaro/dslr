import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import HOUSE_COLORS


def plot_pairwise(df: pd.DataFrame) -> None:
	"""Plot pairwise scatter plots for specified columns in a DataFrame.

	Args:
		df (pd.DataFrame): The DataFrame containing the data.
		columns (list): List of column names to plot.
	"""
	sns.pairplot(
		df,
		hue='Hogwarts House',
		palette=HOUSE_COLORS
	)
	plt.savefig('images/pair_plot.png')
	plt.show()
