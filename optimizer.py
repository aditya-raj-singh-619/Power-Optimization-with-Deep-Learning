# Finds optimal inputs for target power
import numpy as np
from scipy.optimize import minimize
from model import predict


def optimize_features(desired_power, weights, bias, scaler):
    def objective(features):
        features = np.array(features).reshape(1, -1)
        normalized_features = scaler.transform(features)
        prediction = predict(normalized_features, weights, bias)
        return np.abs(desired_power - prediction)

    initial_guess = [0] * len(weights)
    bounds = [(-3, 3)] * len(weights)

    result = minimize(objective, initial_guess, bounds=bounds)
    optimized_features_normalized = result.x.reshape(1, -1)
    optimized_features = scaler.inverse_transform(optimized_features_normalized)

    return optimized_features.flatten()