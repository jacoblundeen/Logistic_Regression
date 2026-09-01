# Logistic Regression

A binary logistic regression classifier implemented from scratch in Python, trained via gradient descent using only NumPy for numerical computation. Built for a graduate Artificial Intelligence course.

## What it does

- **Sigmoid-based prediction** (`calculate_yhat`) — computes the predicted probability for each observation
- **Binary cross-entropy loss** (`calculate_error`) — measures how well the current model fits the data
- **Batch gradient descent** (`learn_model`) — iteratively updates model weights (thetas) to minimize loss, with an adaptive learning rate that shrinks whenever an update makes the error worse, and a convergence threshold that stops training once the error stabilizes
- **Evaluation** (`evaluate`) — reports overall error rate and a confusion matrix (true/false positives and negatives)

## Example Problem (`main.py`)

The included demo is a synthetic terrain-classification problem: given 16 noisy binary sensor readings, classify whether the underlying terrain is "hills" versus one of three other terrain types (plains, forest, swamp). A small set of hand-labeled example readings is expanded into a larger training/test set using a `blur()` function that adds Gaussian noise to each reading, simulating real-world sensor uncertainty.

## Tools

Python, NumPy
