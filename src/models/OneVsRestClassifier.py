from src.utils.header import MODEL_DATA_PATH
from typing import Dict, List
import numpy as np
import json


class OneVsRestClassifier:
	"""One-vs-Rest (OvR) classifier for multi-class classification."""
	def __init__(self, base_model_class):
		"""
		Initializes the OneVsRestClassifier with a base classifier.
		
		Args:
			base_classifier: An instance of a binary classifier (e.g., LogisticRegression).
		"""
		self.base_model_class = base_model_class
		self.classifiers = {} # class_label: model instance
	

	def fit(self, X: np.ndarray, y: np.ndarray) -> None:
		"""
		Fits the OneVsRestClassifier to the training data.
		
		Args:
			X (np.ndarray): Feature matrix of shape (n_samples, n_features).
			y (np.ndarray): Target vector of shape (n_samples,).
		
		Returns:
			None
		"""
		unique_classes = np.unique(y)
		for cls in unique_classes:
			binary_y = (y == cls).astype(int)
			self.classifiers[cls] = self.base_model_class(learning_rate=0.1, iterations=1000, track_cost=False)
			self.classifiers[cls].fit(X, binary_y)

	
	def load_from_weights(self, weights: Dict[str, List[float]]) -> None:
		"""
		Loads the weights for each class classifier.
		
		Args:
			weights (Dict[str, List[float]]): Dictionary mapping class labels to their respective weights.
		
		Returns:
			None
		"""
		for label, weights in weights.items():
			model = self.base_model_class(learning_rate=0.1, iterations=1000, track_cost=False)
			model.load_weights(np.array(weights))
			self.classifiers[label] = model


	def safe_model_to_file(
		self,
		mean: np.ndarray,
		std: np.ndarray,
		filename: str = MODEL_DATA_PATH
	) -> None:
		"""
		Load the weights into a JSON file.

		Args:
			mean (np.ndarray): Array of means for each feature of the training data.
			std (np.ndarray): Array of standard deviations for each feature of the training data.
			filename (str): The path to the JSON file where the weights will be saved.
		
		Returns:
			None
		"""
		model_data = {
			"mean": mean.tolist(),
			"std": std.tolist(),
			"weights": {
				label: clf.weights.tolist()
				for label, clf in self.classifiers.items()
			}
		}
		with open(filename, "w") as f:
			json.dump(model_data, f)
