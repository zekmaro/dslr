import numpy as np
from numpy.typing import NDArray
from typing import List

class LogisticRegression:
    """Logistic Regression model for binary classification."""
    def __init__(
        self,
        learning_rate: float = 0.01,
        iterations: int = 1000,
        track_cost: bool = False,
    ) -> None:
        """
        Initializes the Logistic Regression model
        with a learning rate and number of iterations.

        Args:
            learning_rate (float): The step size for gradient descent.
            iterations (int): The number of iterations for training.
            track_cost (bool): Whether to track the cost during training.
        
        Attributes:
            weights (NDArray[np.float64]): The weights of the model.
            cost_history (List[float]): The history of the cost function if tracking is enabled.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights: NDArray[np.float64] = None
        self.track_cost = track_cost
        self.cost_history = [] if track_cost else None


    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Computes the sigmoid function for the input array."""
        return 1 / (1 + np.exp(-z))


    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Computes the cost function for logistic regression."""
        m = X.shape[0]
        h = self.sigmoid(X @ self.weights)
        return -1/m * np.sum(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8))


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the logistic regression model to the training data using gradient descent."""
        m, n = X.shape
        self.weights = np.zeros(n)
        if self.track_cost:
            self.cost_history = []

        for _ in range(self.iterations):
            h = self.sigmoid(X @ self.weights)
            gradient = (1/m) * (X.T @ (h - y))
            if self.track_cost:
                cost = self.compute_cost(X, y)
                self.cost_history.append(cost)
                if self.track_cost:
                    cost = self.compute_cost(X, y)
                    self.cost_history.append(cost)
            self.weights -= self.learning_rate * gradient
    

    def load_weights(self, weights: List[float]) -> None:
        """Loads the weights into the model."""
        self.weights = np.array(weights)


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts the probabilities of the class."""
        return self.sigmoid(X @ self.weights)
