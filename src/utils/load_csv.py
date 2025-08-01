import pandas as pd


def load(path: str) -> pd.DataFrame:
	"""Load a CSV file into a pandas DataFrame."""
	if not isinstance(path, str):
		raise TypeError("path must be a string")

	try:
		df = pd.read_csv(path)
	except FileNotFoundError:
		raise FileNotFoundError(f"File {path} not found")
	except pd.errors.EmptyDataError:
		raise ValueError(f"File {path} is empty")
	except pd.errors.ParserError:
		raise ValueError(f"File {path} is not a valid CSV file")
	
	# print(f"Loading dataset of dimensions {df.shape}")
	return df
