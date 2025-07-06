import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.header import HOUSE_COLORS, IMAGE_DEST_PATH


def plot_pairwise(df: pd.DataFrame) -> None:
	"""Plot pairwise scatter plots for specified columns in a DataFrame.

	Args:
		df (pd.DataFrame): The DataFrame containing the data.
	"""
	sns.pairplot(
		df,
		hue='Hogwarts House',
		palette=HOUSE_COLORS
	)
	plt.savefig(IMAGE_DEST_PATH)
