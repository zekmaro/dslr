"""
whole thing implemented with scikit
"""

# import sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


DATA_PATH = "./datasets/dataset_train.csv"
CORR_MATRIX_PATH = "./images/correlation_matrix.png"
NAME_DIST_PATH = "./images/name_distribution.png"
SURNAME_DIST_PATH = "./images/surname_distribution.png"


def get_correlations(df: pd.DataFrame):
    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(CORR_MATRIX_PATH, dpi=150)
    plt.close()


def get_name_distribution(df: pd.DataFrame):
    import os
    os.makedirs("./images", exist_ok=True)

    # Name to House distribution
    name_house = pd.crosstab(df['First Name'], df['Hogwarts House'])
    fig, ax = plt.subplots(figsize=(18, 6))
    name_house.plot(kind='bar', stacked=True, ax=ax)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.title('First Name Distribution by House', fontsize=16)
    plt.xlabel('First Name')
    plt.ylabel('Count')
    plt.legend(title='House')
    plt.tight_layout()
    plt.savefig(NAME_DIST_PATH, dpi=150, bbox_inches='tight')
    plt.close()

    # Surname to House distribution
    surname_house = pd.crosstab(df['Last Name'], df['Hogwarts House'])
    fig, ax = plt.subplots(figsize=(18, 6))
    surname_house.plot(kind='bar', stacked=True, ax=ax)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.title('Last Name Distribution by House', fontsize=16)
    plt.xlabel('Last Name')
    plt.ylabel('Count')
    plt.legend(title='House')
    plt.tight_layout()
    plt.savefig(SURNAME_DIST_PATH, dpi=150, bbox_inches='tight')
    plt.close()


def prepare_dataset(path):
    train_data = pd.read_csv(path)

    # Best hand to numerical feat
    hand_mapping = {k: v for v, k in enumerate(train_data['Best Hand'].unique())}
    train_data['Best Hand'] = train_data['Best Hand'].map(hand_mapping)

    get_name_distribution(train_data)
    # get_correlations(train_data)
    return train_data


def main() -> None:
    """Load the trained model and evaluate its accuracy on the test set."""
    df = prepare_dataset(DATA_PATH)


if __name__ == "__main__":
    main()
