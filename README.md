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
* **Evaluation Metrics** - Held-out validation accuracy reported during training
* **Data Visualization** - Detailed plots using `matplotlib` and `seaborn`

## 🧠 Concepts Covered

* Logistic regression
* Sigmoid function & decision boundaries
* Cost function & gradient descent
* Multi-class classification (One-vs-All)
* Model evaluation metrics
* Data scaling and normalization

## 🧰 Requirements

* [`uv`](https://docs.astral.sh/uv/) for dependency & environment management

  * `numpy`
  * `pandas`
  * `matplotlib`
  * `seaborn`

Create the virtual environment and install the locked dependencies:

```sh
uv sync
```

## 🛠️ Usage

### 1. Data Description

```sh
uv run describe datasets/dataset_train.csv
```

* Outputs statistical description: mean, std, min, max, percentiles

### 2. Data Visualization

```sh
uv run histogram datasets/dataset_train.csv
uv run scatter_plot datasets/dataset_train.csv
uv run pair_plot datasets/dataset_train.csv
```

* Histograms by house
* Pairwise feature plots
* Scatter plots between any two features

### 3. Training the Model

```sh
uv run logreg_train datasets/dataset_train.csv
```

* Trains a one-vs-all logistic regression classifier with batch gradient descent
* Holds out a validation split and reports train/validation accuracy
* Writes the weights and the normalization statistics to `shared_data/model.json`
* Hyperparameters are read from `configs/train_config.json`

### 4. Predicting Houses

```sh
uv run logreg_predict datasets/dataset_test.csv shared_data/model.json
```

* Predicts the Hogwarts house for every student in the test set
* Writes `houses.csv` with the `Index,Hogwarts House` schema

## 📁 Project Structure

```
📂 dslr/
├── configs/train_config.json     # Training hyperparameters
├── datasets/                     # dataset_train.csv, dataset_test.csv
├── shared_data/model.json        # Trained weights + normalization stats (generated)
├── src/
│   ├── config.py                 # Train config & model artifact (de)serialization
│   ├── data.py                   # CSV loading, imputation, standardization
│   ├── model.py                  # LogisticRegressionGD + OneVsRestClassifier
│   ├── logreg_train.py           # Training program
│   ├── logreg_predict.py         # Prediction program
│   ├── scripts/                  # describe, histogram, scatter_plot, pair_plot
│   └── utils/                    # Hand-rolled statistics, CSV loader, constants
└── pyproject.toml                # Dependencies & console entry points
```

## 📊 Example Output

```
classes        : ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
features used  : 13
train accuracy : 0.9805  (1280 rows)
val accuracy   : 0.9875  (320 rows held out)
```

## 🏗️ Future Improvements

* Cross-validation
* Support for different optimization algorithms (e.g., SGD, mini-batch GD)
* More robust handling of missing values
* GUI or interactive notebook interface

## 🏆 Credits

* **Developer:** [zekmaro](https://github.com/zekmaro) [vova](https://github.com/vilvl)
* **Project:** Part of the 42 School curriculum
* **Inspiration:** Kaggle-style data science pipelines

---

🔮 May the Sorting Hat be accurate! Explore data, visualize it, and classify away!
