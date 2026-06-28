from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .data import Preprocessor

DEFAULT_CONFIG_PATH = "configs/train_config.json"
DEFAULT_ARTIFACT_PATH = "shared_data/model.json"


@dataclass
class LearningParams:
    learning_rate: float
    epochs: int
    seed: int


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
    classes: list[str]
    weights: dict[str, list[float]] = field(default_factory=dict)

    def save(self, path: str = DEFAULT_ARTIFACT_PATH) -> None:
        payload = {
            "classes": self.classes,
            "preprocessor": self.preprocessor.to_dict(),
            "weights": self.weights,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str = DEFAULT_ARTIFACT_PATH) -> "ModelArtifact":
        payload = json.loads(Path(path).read_text())
        return cls(
            preprocessor=Preprocessor.from_dict(payload["preprocessor"]),
            classes=list(payload["classes"]),
            weights=dict(payload["weights"]),
        )
