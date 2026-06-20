import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.header import LABEL_COLORS, IMAGE_DEST_PATH
from utils.load_csv import load


def plot_pairwise(df: pd.DataFrame) -> None:
    """Plot pairwise scatter plots for specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
    """
    sns.pairplot(
        df,
        hue='Hogwarts House',
        palette=LABEL_COLORS
    )
    plt.savefig(IMAGE_DEST_PATH)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: pair_plot <path_to_csv>")
        sys.exit(1)

    df = load(sys.argv[1])
    bd_col = pd.to_datetime(df['Birthday'])
    df['BD day'] = bd_col.dt.day
    df['BD month'] = bd_col.dt.month
    df['BD year'] = bd_col.dt.year
    df['BD doy'] = bd_col.dt.dayofyear
    df['BD weekday'] = bd_col.dt.weekday
    hand_mapping = {k: v for v, k in enumerate(df['Best Hand'].unique())}
    df['Best Hand'] = df['Best Hand'].map(hand_mapping)
    plot_pairwise(df)


if __name__ == "__main__":
    main()
