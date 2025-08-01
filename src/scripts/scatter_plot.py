import pandas as pd
import matplotlib.pyplot as plt
from utils.header import CORRELATION_THRESHOLD, LABEL_COLORS


def plot_features(data: pd.DataFrame, feature_1: pd.DataFrame, feature_2: pd.DataFrame):
	for house, color in LABEL_COLORS.items():
		subset = data[data["Hogwarts House"] == house]
		plt.scatter(subset[feature_1], subset[feature_2], label=house, color=color, alpha=0.6, s=15)
	plt.xlabel(feature_1)
	plt.ylabel(feature_2)
	plt.title(f"{feature_1} vs {feature_2}")
	plt.legend()
	plt.grid(True)
	plt.show()


def find_similar_features(df: pd.DataFrame):
    """Find similar features in a DataFrame based on mean and standard deviation."""
    num_columns = [col for col in df.columns if df[col].dtype in [int, float]]
    normalized = df[num_columns].apply(lambda col: (col - col.mean()) / col.std())

    similar_features = []
    for i in range(len(num_columns)):
        for j in range(i + 1, len(num_columns)):
            col1 = normalized[num_columns[i]]
            col2 = normalized[num_columns[j]]
            correlation = col1.corr(col2)
            if abs(correlation) > CORRELATION_THRESHOLD:
                similar_features.append((num_columns[i], num_columns[j], correlation))

    print("Similar features based on correlation:")
    for feature1, feature2, corr in similar_features:
        print(f"{feature1} and {feature2} with correlation {corr:.2f}")
        plot_features(df, feature1, feature2)
    return similar_features
