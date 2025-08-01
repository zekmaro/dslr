import matplotlib.pyplot as plt
from typing import List
import numpy as np


def plot_cost_history(
    cost_history: List[float],
    title: str = "Cost History",
    xlabel: str = "Epoch",
    ylable: str = "Cost"
) -> None:
    """
    Plot the cost history of the model during training.

    Args:
        cost_history (List[float]): List of cost values recorded during training.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylable (str): Label for the y-axis.
    
    Returns:
        None
    """
    plt.plot(cost_history)
    plt.xlabel(xlabel)
    plt.ylabel(ylable)
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_heat_map(corr: np.ndarray, title: str = "Correlation Matrix") -> None:
	"""
	Plot a heatmap of the correlation matrix.
    
    Args:
        corr (np.ndarray): The correlation matrix to plot.
        title (str): Title of the heatmap.
    
    Returns:
        None
	"""
	plt.figure(figsize=(10, 8))
	plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
	plt.colorbar()
	plt.title(title)
	plt.show()
