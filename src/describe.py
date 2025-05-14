import sys
import os
import pandas as pd


def describe(df: pd.DataFrame) -> pd.DataFrame:
	"""Describe a pandas DataFrame."""
	if not isinstance(df, pd.DataFrame):
		raise TypeError("df must be a pandas DataFrame")

	described_df = pd.DataFrame()
	described_df["column"] = df.columns
	described_df.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']


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
