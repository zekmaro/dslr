from load_csv import load
from describe import describe


def main():
	df = load("../datasets/dataset_train.csv")
	# print(df.head())
	# print(df.describe())
	# print(df.info())
	# print(df.columns)
	# print()
	describe(df)
	


if __name__ == "__main__":
	main()
