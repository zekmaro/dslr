# dslr - Data Science & Logistic Regression

## 🧪 Project Overview

`dslr` is a data science project from the 42 curriculum that focuses on applying machine learning concepts to real datasets. The project involves building tools to explore data, visualize it, and apply logistic regression to perform classification — specifically, predicting Hogwarts house placement from student data.

This project serves as an introduction to core machine learning concepts such as feature scaling, training/testing datasets, logistic regression, and model evaluation, all implemented from scratch in Python.

## 🚀 Features

* **CSV Data Parsing** - Manual loading and preprocessing of CSV files
* **Data Exploration** - Statistical summaries and visualizations (histograms, scatter plots, pair plots)
* **Feature Normalization** - Scaling features for optimal gradient descent performance
* **Logistic Regression Classifier** - One-vs-All strategy for multi-class classification
* **Training & Prediction** - Train a model and use it to predict classes on unseen data
* **Evaluation Metrics** - Accuracy, precision, recall, F1-score, confusion matrix
* **Data Visualization** - Detailed plots using `matplotlib` and `seaborn`

## 🧠 Concepts Covered

* Logistic regression
* Sigmoid function & decision boundaries
* Cost function & gradient descent
* Multi-class classification (One-vs-All)
* Model evaluation metrics
* Data scaling and normalization

## 🧰 Requirements

* Python 3.x
* Libraries:

  * `numpy`
  * `pandas`
  * `matplotlib`
  * `seaborn`
  * (optional) `scikit-learn` for metrics comparison

Install dependencies:

```sh
pip install -r requirements.txt
```

## 🛠️ Usage

### 1. Data Description

```sh
python describe.py dataset.csv
```

* Outputs statistical description: mean, std, min, max, percentiles

### 2. Data Visualization

```sh
python histogram.py dataset.csv
python scatter_plot.py dataset.csv
python pair_plot.py dataset.csv
```

* Histograms by house
* Pairwise feature plots
* Scatter plots between any two features

### 3. Training the Model

```sh
python logreg_train.py dataset_train.csv
```

* Trains logistic regression model for multi-class classification
* Stores trained weights to a file (e.g., `weights.npy`)

### 4. Predicting Houses

```sh
python logreg_predict.py dataset_test.csv
```

* Predicts Hogwarts house for each student in test data
* Outputs `houses.csv`

### 5. Evaluating Model

```sh
python evaluate.py dataset_test.csv houses.csv
```

* Displays accuracy, precision, recall, F1-score, and confusion matrix

## 📁 Project Structure

```
📂 dslr/
├── describe.py          # Statistical summary of dataset
├── histogram.py         # Data visualization by class
├── scatter_plot.py      # Feature scatter plots
├── pair_plot.py         # Seaborn pair plots
├── logreg_train.py      # Model training logic
├── logreg_predict.py    # Model inference/prediction
├── evaluate.py          # Metrics and model evaluation
├── requirements.txt     # Dependencies
├── weights.npy          # Saved model weights
```

## 📊 Example Output

```
Training accuracy: 91.2%
Precision per class: [0.89, 0.93, 0.92, 0.90]
F1 Score: 0.91
```

## 🏗️ Future Improvements

* Cross-validation
* Support for different optimization algorithms (e.g., SGD, Adam)
* More robust handling of missing values
* GUI or interactive notebook interface

## 🏆 Credits

* **Developer:** [zekmaro](https://github.com/zekmaro)
* **Project:** Part of the 42 School curriculum
* **Inspiration:** Kaggle-style data science pipelines

---

🔮 May the Sorting Hat be accurate! Explore data, visualize it, and classify away!
