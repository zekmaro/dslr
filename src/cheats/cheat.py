"""
whole thing implemented with scikit
"""

# import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DATA_PATH = "./datasets/dataset_train.csv"
CORR_MATRIX_PATH = "./images/correlation_matrix.png"
NAME_DIST_PATH = "./images/name_distribution.png"
SURNAME_DIST_PATH = "./images/surname_distribution.png"


def get_correlations(df: pd.DataFrame):
    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(15, 10))
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

def add_bd_cols(df: pd.DataFrame):
    bd_col = pd.to_datetime(df['Birthday'])
    df['BD day'] = bd_col.dt.day
    df['BD month'] = bd_col.dt.month
    df['BD year'] = bd_col.dt.year
    df['BD doy'] = bd_col.dt.dayofyear
    df['BD weekday'] = bd_col.dt.weekday

def prepare_dataset(path):
    train_data = pd.read_csv(path)

    # Best hand to numerical feat
    hand_mapping = {k: v for v, k in enumerate(train_data['Best Hand'].unique())}
    train_data['Best Hand'] = train_data['Best Hand'].map(hand_mapping)

    # get_name_distribution(train_data)
    add_bd_cols(train_data)
    get_correlations(train_data)

    # train_data = train_data.drop("Defense Against the Dark Arts")  # is correlated with Astronomy

    return train_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris

def main() -> None:
    """Load the trained model and evaluate its accuracy on the test set."""
    df = prepare_dataset(DATA_PATH)

    y = df['Hogwarts House']
    # X = df[['Best Hand', '']]


# ---------------------------------------------------
# Train / Test Split
# ---------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# # ---------------------------------------------------
# # Pipeline
# # ---------------------------------------------------
# pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     (
#         "classifier",
#         OneVsRestClassifier(
#             LogisticRegression(
#                 max_iter=5000,
#                 solver="lbfgs"
#             )
#         )
#     )
# ])

# # ---------------------------------------------------
# # Train
# # ---------------------------------------------------
# pipeline.fit(X_train, y_train)

# # ---------------------------------------------------
# # Predict
# # ---------------------------------------------------
# y_pred = pipeline.predict(X_test)

# # ---------------------------------------------------
# # Evaluation
# # ---------------------------------------------------
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n")
# print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
