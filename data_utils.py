 # Data Loading and Preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    required_columns = ["Temperature", "MachineUtilization", "ProductionRate", "Humidity", "PowerConsumption"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' is missing from the dataset.")

    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data.fillna(data.mean(), inplace=True)

    if data.empty:
        raise ValueError("The dataset is empty after preprocessing. Check your CSV file for valid data.")

    X = data[["Temperature", "MachineUtilization", "ProductionRate", "Humidity"]].values
    y = data["PowerConsumption"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler

# Contains prediction, cost, gradient descent

