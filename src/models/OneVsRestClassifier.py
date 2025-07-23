from src.utils.header import MODEL_DATA_PATH, LABEL_MAP
from typing import Dict, List
import pandas as pd
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
		self.feature_means = None
		self.feature_stds = None
		self.feature_names = None
	

	def fit(
		self,
		X: np.ndarray,
		y: np.ndarray,
		feature_names: List[str]
	) -> None:
		"""
		Fits the OneVsRestClassifier to the training data.
		
		Args:
			X (np.ndarray): Feature matrix of shape (n_samples, n_features).
			y (np.ndarray): Target vector of shape (n_samples,).
		
		Returns:
			None
		"""
		self.feature_names = feature_names
		X = X[feature_names]
		self.feature_means = X.mean(axis=0)
		self.feature_stds = X.std(axis=0)
		normalized_X = (X - self.feature_means) / self.feature_stds
		unique_classes = np.unique(y)
		for cls in unique_classes:
			binary_y = (y == cls).astype(int)
			self.classifiers[cls] = self.base_model_class(learning_rate=0.1, iterations=1000, track_cost=False)
			self.classifiers[cls].fit(normalized_X, binary_y)

	
	def load_from_weights(
		self,
		weights: Dict[str, List[float]]
	) -> None:
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


	def save_model_to_file(
		self,
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
			"mean": self.feature_means.tolist(),
			"std": self.feature_stds.tolist(),
			"weights": {
				label: clf.weights.tolist()
				for label, clf in self.classifiers.items()
			}
		}
		with open(filename, "w") as f:
			json.dump(model_data, f)


	def normalize_data(
		self,
		x_clean: pd.DataFrame,
		feature_names: List[str],
	) -> np.ndarray:
		"""
		Normalize the student data by calculating the mean and standard deviation.

		Args:
			x_clean (pd.DataFrame): The cleaned training data.
			feature_means (np.ndarray): The means of the features.
			feature_stds (np.ndarray): The standard deviations of the features.
		
		Returns:
			np.ndarray: The normalized student data.
		"""
		if self.feature_means is None or self.feature_stds is None:
			raise ValueError("Feature means and standard deviations must be set before normalization.")
		features_data = x_clean[feature_names].to_numpy()
		return (features_data - self.feature_means) / self.feature_stds
	

	def load_model_from_file(
		self,
		filename: str = MODEL_DATA_PATH
	) -> None:
		"""
		Load the model weights from a JSON file.
		
		Args:
			filename (str): The path to the JSON file containing the model weights.
		
		Returns:
			None
		"""
		with open(filename, "r") as f:
			model_data = json.load(f)
		self.feature_means = np.array(model_data["mean"])
		self.feature_stds = np.array(model_data["std"])
		self.load_from_weights(model_data["weights"])


	def predict(
		self,
		normalized_data: np.ndarray
	) -> List[str]:
		"""
		Predict the class label for each row in the normalized dataset.

		Args:
			normalized_data (np.ndarray): The normalized feature data.

		Returns:
			List[str]: The predicted class labels for each sample.
		"""
		predicted_labels = []

		for row in normalized_data:
			probabilities = {}
			for label, classifier in self.classifiers.items():
				probabilities[label] = classifier.predict_proba(row)
			best_label = max(probabilities, key=probabilities.get)  # type: ignore
			predicted_labels.append(best_label)

		return predicted_labels


	def evaluate(
		self,
		X: pd.DataFrame,
		y: pd.Series,
		feature_names: List[str]
	) -> float:
		"""
		Calculate the accuracy of the model on the given dataset.
		
		Args:
			X (pd.DataFrame): Feature matrix of shape (n_samples, n_features).
			y (pd.Series): Target vector of shape (n_samples,).
			feature_names (List[str]): List of feature names to use for prediction.
		
		Returns:
			float: The accuracy of the model on the dataset.
		"""
		X_normalized = self.normalize_data(X, feature_names)
		predictions = self.predict(X_normalized)
		predicted_labels = np.array([LABEL_MAP[p] for p in predictions])
		true_labels = y.map(LABEL_MAP).to_numpy()
		return np.mean(predicted_labels == true_labels)
