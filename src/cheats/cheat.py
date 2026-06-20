"""
whole thing implemented with scikit
"""

# import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance


DATA_PATH = "./datasets/dataset_train.csv"
CORR_MATRIX_PATH = "./images/correlation_matrix.png"
NAME_DIST_PATH = "./images/name_distribution.png"
SURNAME_DIST_PATH = "./images/surname_distribution.png"
FEATURE_IMPORTANCE_PATH = "./images/feature_importance.png"
FEATURE_IMPACT_CONFIG_PATH = "./configs/df_config.json"


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


def analyze_feature_impact(pipeline, X_test, y_test, feature_names):
    """Analyze and visualize feature impact on model predictions."""
    import os
    os.makedirs("./configs", exist_ok=True)
    os.makedirs("./images", exist_ok=True)

    classifier = pipeline.named_steps['classifier']

    # Stack coefficients from all per-class estimators, shape (n_classes, n_features)
    all_coefficients = np.vstack([est.coef_[0] for est in classifier.estimators_])
    coef_importance = np.abs(all_coefficients).mean(axis=0)

    # Create dataframe for coefficient importance
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coef_importance
    }).sort_values('Importance', ascending=False)

    # Compute permutation importance
    perm_importance = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    perm_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)

    # Visualize both importance metrics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Coefficient importance
    top_n = 15
    coef_top = coef_df.head(top_n)
    axes[0].barh(coef_top['Feature'], coef_top['Importance'], color='steelblue')
    axes[0].set_xlabel('Mean Absolute Coefficient')
    axes[0].set_title('Feature Importance (Model Coefficients)', fontsize=14)
    axes[0].invert_yaxis()

    # Permutation importance
    perm_top = perm_df.head(top_n)
    axes[1].barh(perm_top['Feature'], perm_top['Importance'],
                 xerr=perm_top['Std'], color='coral', capsize=3)
    axes[1].set_xlabel('Permutation Importance')
    axes[1].set_title('Feature Impact (Permutation Importance)', fontsize=14)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150, bbox_inches='tight')
    plt.close()

    # Save impact analysis to config
    impact_config = {
        'coefficient_importance': coef_df.to_dict('records'),
        'permutation_importance': perm_df.to_dict('records'),
        'top_5_features': coef_df.head(5)['Feature'].tolist()
    }

    with open(FEATURE_IMPACT_CONFIG_PATH, 'w') as f:
        json.dump(impact_config, f, indent=2)

    print("\n" + "="*60)
    print("FEATURE IMPACT ANALYSIS")
    print("="*60)
    print("\nTop 10 Features by Model Coefficients:")
    print(coef_df.head(10).to_string(index=False))
    print("\nTop 10 Features by Permutation Importance:")
    print(perm_df.head(10).to_string(index=False))
    print(f"\nVisualization saved to: {FEATURE_IMPORTANCE_PATH}")
    print(f"Config saved to: {FEATURE_IMPACT_CONFIG_PATH}")
    print("="*60 + "\n")


def prepare_dataset(path):
    df = pd.read_csv(path)

    # Best hand to numerical feat
    hand_mapping = {k: v for v, k in enumerate(df['Best Hand'].unique())}
    df['Best Hand'] = df['Best Hand'].map(hand_mapping)

    # get_name_distribution(train_data)
    add_bd_cols(df)
    # get_correlations(train_data)
    df = df.dropna()

    # train_data = train_data.drop("Defense Against the Dark Arts")  # is correlated with Astronomy

    return df


def main() -> None:
    """Load the trained model and evaluate its accuracy on the test set."""
    df = prepare_dataset(DATA_PATH)

    PREDICT_FEAT = 'Hogwarts House'
    ALL_FEATS = [
        # "Best Hand",
        # "Arithmancy",
        "Astronomy",
        "Herbology",
        # "Defense Against the Dark Arts",
        # "Divination",
        # "Muggle Studies",
        "Ancient Runes",
        # "History of Magic",
        # "Transfiguration",
        # "Potions",
        # "Care of Magical Creatures",
        # "Charms",
        # "Flying",
        # 'BD day',
        # 'BD month',
        # 'BD year',
        # 'BD doy',
        # 'BD weekday'
    ]
    y = df[PREDICT_FEAT]
    X = df[ALL_FEATS]
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=5000,
                    solver='liblinear'
                )
            )
        )
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Analyze feature impact
    analyze_feature_impact(pipeline, X_test, y_test, ALL_FEATS)


if __name__ == "__main__":
    main()
