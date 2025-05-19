import matplotlib.pyplot as plt
import numpy as np


def plot_heat_map(corr: np.ndarray, title: str = "Correlation Matrix") -> None:
	"""
	Plot a heatmap of the correlation matrix.
	:param corr: Correlation matrix
	:param title: Title of the plot
	"""
	plt.figure(figsize=(10, 8))
	plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
	plt.colorbar()
	plt.title(title)
	plt.show()
