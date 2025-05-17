import sys
import os
import pandas as pd


def describe(df: pd.DataFrame) -> pd.DataFrame:
	"""Describe a pandas DataFrame."""
	if not isinstance(df, pd.DataFrame):
		raise TypeError("df must be a pandas DataFrame")

	num_columns = [col for col in df.columns if df[col].dtype in [int, float]]

	described_df = pd.DataFrame(
		index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
		columns=num_columns
	)

	for col in df.columns:
		dtype = df[col].dtype

		if dtype == int or dtype == float:
			described_df[col] = [
				df[col].count(),
				df[col].mean(),
				df[col].std(),
				df[col].min(),
				df[col].quantile(0.25),
				df[col].median(),
				df[col].quantile(0.75),
				df[col].max(),
			]

	print(described_df)


def main():
	if len(sys.argv) != 2:
		print("Usage: python describe.py <path_to_csv>")
		sys.exit(1)
		
	path = sys.argv[1]

	if not os.path.isfile(path):
		print(f"File {path} not found")
		sys.exit(1)

	df = pd.read_csv(path)
	described_df = describe(df)
	print(described_df)

if __name__ == "__main__":
	main()
