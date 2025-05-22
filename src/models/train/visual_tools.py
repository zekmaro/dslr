import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
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


def plot_graph(corr_matrix: pd.DataFrame, title: str = "Correlation Graph") -> None:
	"""
	Plot a graph of the correlation matrix.
	:param corr: Correlation matrix
	:param title: Title of the plot
	"""
	G = nx.from_pandas_adjacency(corr_matrix)
	nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
	plt.title(title)
	plt.show()
