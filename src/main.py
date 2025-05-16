from load_csv import load
from describe import describe
from histogram import get_course_scores_per_house


def main():
	df = load("../datasets/dataset_train.csv")
	print(df.head())
	print(df.describe())
	print(df.info())
	print(df.columns)
	# print()
	# describe(df)
	get_course_scores_per_house(df)
	


if __name__ == "__main__":
	main()
