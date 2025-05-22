import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.statistical_methods import calculate_mean, calculate_median, calculate_quartile, calculate_variance, calculate_stddev


def plot_feature_distribution(df: pd.DataFrame, feature: str) -> None:
	"""
	Plot the distribution of a feature.
	:param df: DataFrame
	:param feature: Feature to plot
	"""
	plt.figure(figsize=(8, 4))
	sns.histplot(data=df, x=feature, hue='Hogwarts House', kde=True, element='step')
	plt.title(f'Distribution of {feature} by House')
	plt.xlabel(feature)
	plt.ylabel('Frequency')
	plt.grid(True)
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

	homo_courses = []
	for course, houses in courses_score_per_house.items():
		print(f"Course: {course}")
		counter = 0
		for house, scores in houses.items():
			mean = calculate_mean(*scores)
			std = calculate_stddev(*scores)
			cv = std / mean if mean != 0 else 0
			print(f"\t{house}: mean is {mean}, std is {std}, CV is {cv}")
			if cv < 0.2:
				counter += 1
			if counter == 4:
				homo_courses.append(course)

	print(homo_courses)
	for course in homo_courses:
		plot_feature_distribution(df, course)
