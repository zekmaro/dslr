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
    Plot a graph of the correlation matrix with edge weights shown by width and color.
    """
    G = nx.from_pandas_adjacency(corr_matrix)

    pos = nx.spring_layout(G, seed=42)  # deterministic layout

    # Extract edge weights
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Draw nodes and edges with varying thickness and color
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color='lightblue',
        edge_color=weights,              # color based on weight
        width=[abs(w) * 5 for w in weights],  # thickness scaled by weight
        edge_cmap=plt.cm.plasma,
        node_size=500,
        font_size=10
    )

    # Optionally add edge labels (correlation values)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.tight_layout()
    plt.show()
