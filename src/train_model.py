from __future__ import annotations

import argparse

import numpy as np

from .config import (DEFAULT_ARTIFACT_PATH, DEFAULT_CONFIG_PATH, ModelArtifact,
                     TrainConfig)
from .data import Preprocessor, default_feature_columns, load_dataset
from .model import OneVsRestClassifier


def train_validation_split(n: int, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    return perm[n_val:], perm[:n_val]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the house classifier.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT_PATH)
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config)

    df = load_dataset(cfg.dataset)
    if cfg.feature_columns is None:
        cfg.feature_columns = default_feature_columns(df)

    # drop only rows with no target
    df = df.dropna(subset=[cfg.target_column]).reset_index(drop=True)

    train_idx, val_idx = train_validation_split(len(df), cfg.validation_split, cfg.seed)
    df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]

    # fit on train, then apply to val
    pre = Preprocessor(
        feature_columns=cfg.feature_columns,
        impute_strategy=cfg.impute_strategy,
    )
    x_train = pre.fit_transform(df_train)  # saves statistics in preprocessor
    y_train = df_train[cfg.target_column].to_numpy()

    clf = OneVsRestClassifier(cfg.learning_params()).fit(x_train, y_train)

    train_acc = accuracy(y_train, clf.predict(x_train))
    print(f"classes        : {clf.classes}")
    print(f"features used  : {len(cfg.feature_columns)}")
    print(f"train accuracy : {train_acc:.4f}  ({len(df_train)} rows)")
    if len(df_val):
        x_val = pre.transform(df_val)
        y_val = df_val[cfg.target_column].to_numpy()
        print(f"val accuracy   : {accuracy(y_val, clf.predict(x_val)):.4f}"
              f"  ({len(df_val)} rows held out)")

    ModelArtifact(
        preprocessor=pre,
        classifier=clf,
    ).save(args.artifact)
    print(f"\nsaved recipe   -> {args.config}")
    print(f"saved artifact -> {args.artifact}")


if __name__ == "__main__":
    main()
