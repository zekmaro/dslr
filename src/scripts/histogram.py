import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.statistical_methods import calculate_mean, calculate_median, calculate_quartile, calculate_variance, calculate_stddev
from utils.header import THRESHOLD_MEAN_CV, THRESHOLD_STD_CV


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

    df = df.copy()
    for course in courses:
        mean = df[course].mean()
        std = df[course].std()
        if std != 0:
            df[course] = (df[course] - mean) / std
            course = df[course]

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
    for course, houses_data in courses_score_per_house.items():
        means = [calculate_mean(*houses_data[house]) for house in houses_data]
        stddevs = [calculate_stddev(*houses_data[house]) for house in houses_data]
        
        print(f"Course's means: {means}")
        print(f"Course's stds: {stddevs}")

        means_cv = calculate_stddev(*means)/abs(calculate_mean(*means))
        stds_cv = calculate_stddev(*stddevs)/abs(calculate_mean(*stddevs))
        print(f"Course: {course}. means_cv: {means_cv}, stds_cv: {stds_cv}")
        if means_cv < THRESHOLD_MEAN_CV and stds_cv < THRESHOLD_STD_CV:
            homo_courses.append(course)
        
        print()

    print(homo_courses)
    for course in homo_courses:
        plot_feature_distribution(df, course)
