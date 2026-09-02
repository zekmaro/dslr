import sys
import os
import pandas as pd
from ..utils.statistical_methods import (
    count,
    calculate_mean,
    calculate_median,
    calculate_quartile,
    calculate_min,
    calculate_max,
    calculate_stddev
)


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Describe a pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    num_columns = [col for col in df.columns if df[col].dtype in [int, float]]

    described_df = pd.DataFrame(
        index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
        columns=num_columns
    )

    for col in num_columns:
        clean_col = df[col].dropna()
        quart_tuple = calculate_quartile(*clean_col)
        described_df[col] = [
            count(*clean_col),
            calculate_mean(*clean_col),
            calculate_stddev(*clean_col),
            calculate_min(*clean_col),
            quart_tuple[0],
            calculate_median(*clean_col),
            quart_tuple[1],
            calculate_max(*clean_col),
        ]

    return described_df


def main():
    if len(sys.argv) != 2:
        print("Usage: describe <path_to_csv>")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.isfile(path):
        print(f"File {path} not found")
        sys.exit(1)

    df = pd.read_csv(path)
    print(describe(df))


if __name__ == "__main__":
    main()
