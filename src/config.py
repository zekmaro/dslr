from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import Preprocessor
from .model import LearningParams, OneVsRestClassifier


DEFAULT_CONFIG_PATH = "configs/train_config.json"
DEFAULT_ARTIFACT_PATH = "shared_data/model.json"


@dataclass
class TrainConfig:
    dataset: str = "datasets/dataset_train.csv"
    target_column: str = "Hogwarts House"
    feature_columns: list[str] | None = None
    impute_strategy: str = "median"
    learning_rate: float = 0.5
    epochs: int = 1000
    seed: int = 42
    validation_split: float = 0.2

    def learning_params(self) -> LearningParams:
        return LearningParams(
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            seed=self.seed,
        )

    def save(self, path: str = DEFAULT_CONFIG_PATH) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str = DEFAULT_CONFIG_PATH) -> "TrainConfig":
        data = json.loads(Path(path).read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ModelArtifact:
    preprocessor: Preprocessor
    classifier: OneVsRestClassifier

    def save(self, path: str = DEFAULT_ARTIFACT_PATH) -> None:
        payload = {
            "preprocessor": self.preprocessor.to_dict(),
            "classifier": self.classifier.to_dict(),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str = DEFAULT_ARTIFACT_PATH) -> "ModelArtifact":
        payload = json.loads(Path(path).read_text())
        return cls(
            preprocessor=Preprocessor.from_dict(payload["preprocessor"]),
            classifier=OneVsRestClassifier.from_dict(payload["classifier"]),
        )
