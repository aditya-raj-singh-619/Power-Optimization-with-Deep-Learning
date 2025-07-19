# main.py
import numpy as np
from data_utils import load_data
from model import predict, gradient_descent
from optimizer import optimize_features

if __name__ == "__main__":
    file_path = "data_new.csv"

    X, y, scaler = load_data(file_path)

    weights = np.zeros(X.shape[1])
    bias = 0.0
    learning_rate = 0.01
    iterations = 1000

    weights, bias, cost_history = gradient_descent(X, y, weights, bias, learning_rate, iterations)

    print("\nFinal weights:", weights)
    print("Final bias:", bias)

    predictions = predict(X, weights, bias)

    if np.any(y == 0):
        raise ValueError("Target values (PowerConsumption) contain zero. Adjust your data to avoid division by zero.")

    efficiency = 100 - (np.mean(np.abs(predictions - y) / y) * 100)
    print(f"\nModel efficiency: {efficiency:.2f}%")

    new_data = np.array([
        [45.0, 84, 120.0, 60.0],
        [35.0, 65, 100.0, 55.0]
    ])

    scaled_data = scaler.transform(new_data)
    new_predictions = predict(scaled_data, weights, bias)

    for i, prediction in enumerate(new_predictions):
        print(f"Prediction for input {i + 1}: {prediction:.2f} units of power consumption")

    desired_power = 150.0
    optimized_features = optimize_features(desired_power, weights, bias, scaler)

    print(f"\nOptimized features for {desired_power} units of power consumption:")
    print(f"Temperature: {optimized_features[0]:.2f}")
    print(f"MachineUtilization: {optimized_features[1]:.2f}")
    print(f"ProductionRate: {optimized_features[2]:.2f}")
    print(f"Humidity: {optimized_features[3]:.2f}")