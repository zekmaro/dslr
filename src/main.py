from utils.header import TRAIN_DATASET_PATH
from utils.load_csv import load
from scripts.describe import describe
from scripts.histogram import get_course_scores_per_house
from scripts.scatter_plot import find_similar_features
from scripts.pair_plot import plot_pairwise


def main():
	df = load(TRAIN_DATASET_PATH)
	# print(df.head())
	# print(df.info())
	print(df.describe())
	describe(df)
	get_course_scores_per_house(df)
	# print(find_similar_features(df))
	# # plot_pairwise(df)


if __name__ == "__main__":
	main()
