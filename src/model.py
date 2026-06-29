from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class LearningParams:
    """Gradient-descent hyperparameters consumed by `LogisticRegressionGD`."""
    learning_rate: float
    epochs: int
    seed: int


# may overflow on large negative weights, though not expected on normalized data
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


class LogisticRegressionGD:
    def __init__(
        self,
        learning_params: LearningParams | None = None
    ):
        self.learning_params: LearningParams | None = learning_params
        self.theta: np.ndarray | None = None
        self.loss_history: List[float] | None = None

    @classmethod
    def from_weights(cls, theta: np.ndarray):
        obj = cls()
        obj.theta = theta
        return obj

    @staticmethod
    def _add_bias(x: np.ndarray) -> np.ndarray:
        """Prepend a column of ones so theta[0] acts as the intercept"""
        return np.hstack([np.ones((x.shape[0], 1)), x])

    @staticmethod
    def _log_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """Computes the average binary cross-entropy loss function for logistic regression."""
        m = X.shape[0]
        h = _sigmoid(X @ theta)
        eps = 1e-8
        return -1 / m * np.sum(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))

    @staticmethod
    def _log_loss_grad(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        return X.T @ (y_pred - y) / len(y)

    def fit(self, x: np.ndarray, y: np.ndarray, track_loss: bool = False) -> "LogisticRegressionGD":
        if self.learning_params is None:
            raise ValueError("learning_params are not set")

        # rng = np.random.default_rng(self.learning_params.seed)
        xb = self._add_bias(x)
        m, n = xb.shape
        theta = np.zeros(n)

        if track_loss:
            self.loss_history = []

        for _ in range(self.learning_params.epochs):
            # idx = rng.permutation(m)   # shuffle each epoch for stochastic gd later?
            xi, yi = xb, y
            predictions = _sigmoid(xi @ theta)
            grad = self._log_loss_grad(xi, yi, predictions)
            theta -= self.learning_params.learning_rate * grad
            if track_loss:
                self.loss_history.append(self._log_loss(xi, yi, theta))

        self.theta = theta
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probability that each row belongs to the positive class."""
        if self.theta is None:
            raise RuntimeError("model is not fitted")
        return _sigmoid(self._add_bias(x) @ self.theta)


class OneVsRestClassifier:
    """Multiclass classification by training one binary model per class.

    For each house we fit a `LogisticRegressionGD` whose positive class is "this
    house" and whose negative class is "every other house". At prediction time
    every model reports how confident it is, and we take the argmax. Storing the
    class order explicitly is what lets us map that argmax back to a house name.
    """

    def __init__(self, learning_params: LearningParams | None = None):
        # Hyperparameters are forwarded unchanged to each per-class model, so the
        # OVR layer knows nothing about gradient descent — it only orchestrates.
        self.learning_params = learning_params
        self.classes: list[str] = []
        self.models: dict[str, LogisticRegressionGD] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OneVsRestClassifier":
        if self.learning_params is None:
            raise ValueError("Learning params are not set")
        # Sorted for determinism: the artifact is identical across runs.
        self.classes = sorted(np.unique(y).tolist())
        for cls in self.classes:
            binary_target = (y == cls).astype(float)
            self.models[cls] = LogisticRegressionGD(self.learning_params).fit(x, binary_target)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return an (m, n_classes) matrix of per-class probabilities, columns
        ordered like `self.classes`."""
        return np.column_stack([self.models[c].predict_proba(x) for c in self.classes])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the most confident class label for each row."""
        proba = self.predict_proba(x)
        winners = np.argmax(proba, axis=1)
        return np.array([self.classes[i] for i in winners])

    def to_dict(self) -> dict:
        return {
            "classes": self.classes,
            "weights": {c: self.models[c].theta.tolist() for c in self.classes},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OneVsRestClassifier":
        obj = cls()
        obj.classes = list(d["classes"])
        for c in obj.classes:
            obj.models[c] = LogisticRegressionGD.from_weights(
                np.asarray(d["weights"][c], dtype=float)
            )
        return obj
