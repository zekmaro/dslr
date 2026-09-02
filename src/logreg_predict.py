from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DEFAULT_ARTIFACT_PATH, ModelArtifact
from .data import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Hogwarts houses.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="datasets/dataset_test.csv",
        help="set to classify",
    )
    parser.add_argument(
        "weights",
        nargs="?",
        default=DEFAULT_ARTIFACT_PATH,
        help="model file produced by logreg_train",
    )
    parser.add_argument("--out", default="houses.csv")
    args = parser.parse_args()

    artifact = ModelArtifact.load(args.weights)
    clf = artifact.classifier

    df = load_dataset(args.dataset)
    x = artifact.preprocessor.transform(df)
    predictions = clf.predict(x)

    # Output schema is: "Index", "Hogwarts House"
    index = df["Index"] if "Index" in df.columns else range(len(df))
    out = pd.DataFrame({"Index": index, "Hogwarts House": predictions})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"wrote {len(out)} predictions -> {args.out}")
    counts = out["Hogwarts House"].value_counts().to_dict()
    print("predicted house distribution:", counts)


if __name__ == "__main__":
    main()
