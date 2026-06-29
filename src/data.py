from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


NON_FEATURE_COLUMNS = (
    "Index",
)


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def default_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric if c not in NON_FEATURE_COLUMNS]


@dataclass
class Preprocessor:
    feature_columns: list[str]
    impute_strategy: str = "median"  # "median" or "mean"

    # frozen statistics, needed for n/a predict later
    impute_values: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    # calc statistics + None strategy
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        df = df[self.feature_columns]
        means = df.mean()
        stds = df.std()

        if self.impute_strategy == "median":
            fills = df.median()
        elif self.impute_strategy == "mean":
            fills = means
        else:
            raise ValueError(f"unknown impute_strategy {self.impute_strategy}")

        self.stds = stds.to_dict()
        self.means = means.to_dict()
        self.impute_values = fills.to_dict()

        return self

    # normalization on a fitted preprocessor
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_columns].copy()
        x = x.fillna(self.impute_values)
        for col in self.feature_columns:
            x[col] = (x[col] - self.means[col]) / self.stds[col]
        return x.to_numpy(dtype=float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def to_dict(self) -> dict:
        return {
            "feature_columns": self.feature_columns,
            "impute_strategy": self.impute_strategy,
            "impute_values": self.impute_values,
            "means": self.means,
            "stds": self.stds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        return cls(
            feature_columns=list(d["feature_columns"]),
            impute_strategy=d["impute_strategy"],
            impute_values=dict(d["impute_values"]),
            means=dict(d["means"]),
            stds=dict(d["stds"]),
        )
