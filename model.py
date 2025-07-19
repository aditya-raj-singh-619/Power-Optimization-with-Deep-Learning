# Contains prediction, cost, gradient descent

import numpy as np


def predict(X, weights, bias):
    return np.dot(X, weights) + bias


def compute_cost(X, y, weights, bias):
    m = len(y)
    predictions = predict(X, weights, bias)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


def gradient_descent(X, y, weights, bias, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = predict(X, weights, bias)
        dw = (1 / m) * np.dot(X.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        cost = compute_cost(X, y, weights, bias)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}, Weights = {weights}, Bias = {bias}")

    return weights, bias, cost_history