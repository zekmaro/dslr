import pandas as pd
import matplotlib.pyplot as plt
from statistical_methods import calculate_mean

def plot_histogram(df: pd.DataFrame, column: str, bins: int = 10) -> None:
	"""Plot a histogram of a specified column in a DataFrame.

	Args:
		df (pd.DataFrame): The DataFrame containing the data.
		column (str): The column name to plot.
		bins (int, optional): The number of bins for the histogram. Defaults to 10.
	"""
	if column not in df.columns:
		raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

	plt.figure(figsize=(10, 6))
	plt.hist(df[column], bins=bins, edgecolor='black')
	plt.title(f"Histogram of {column}")
	plt.xlabel(column)
	plt.ylabel("Frequency")
	plt.grid(axis='y', alpha=0.75)
	plt.show()


def get_course_scores_per_house(df: pd.DataFrame) -> None:
	"""Find which Hogwarts course has a homogeneous score distribution between all four houses"""
	courses = [col for col in df.columns if df[col].dtype in [int, float]]

	courses_score_per_house = {
		course: {
			"Ravenclaw": [],
			"Hufflepuff": [],
			"Gryffindor": [],
			"Slytherin": []
		} for course in courses
	}

	for col in df.columns:
		if col in courses:
			for _, row in df.iterrows():
				if pd.notna(row[col]):
					courses_score_per_house[col][row["Hogwarts House"]].append(row[col])

	for course, houses in courses_score_per_house.items():
		print(f"Course: {course}")
		for house, scores in houses.items():
			print(f"\t{house}: {calculate_mean(*scores)}")
			# print(f"\t{house}: {scores}")
