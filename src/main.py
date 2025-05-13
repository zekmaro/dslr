from load_csv import load


def main():
	df = load("../datasets/dataset_train.csv")
	print(df.head())
	print(df.describe())
	print(df.info())
	print(df.columns)


if __name__ == "__main__":
	main()