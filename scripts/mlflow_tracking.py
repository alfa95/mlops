import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from river.drift import ADWIN  # ✅ Replaced skmultiflow with river
import os
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

# ==========================
# Load & Preprocess Data
# ==========================

# Load Fashion MNIST and reduce dataset size for efficiency
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, y_train = x_train[:10000], y_train[:10000]  # Reduce to 10K samples
x_test, y_test = x_test[:2000], y_test[:2000]  # Reduce test set for efficiency

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ==========================
# MLflow Experiment Setup
# ==========================

mlflow.set_tracking_uri("./mlruns")  # ✅ Save logs locally
mlflow.set_experiment("FashionMNIST-Tracking")

# Initialize ADWIN for drift detection across runs
adwin = ADWIN()
previous_acc = None

# ==========================
# Train & Log Model in MLflow
# ==========================
for run in range(5):  # Train 5 times to observe drift
    with mlflow.start_run():
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), verbose=1)

        # Evaluate model
        test_loss, test_acc = model.evaluate(x_test, y_test)

        # Log metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        # Log model
        mlflow.keras.log_model(model, "model")

        print(f"Run {run+1}: Accuracy = {test_acc:.4f}, Loss = {test_loss:.4f}")

        # Add accuracy to ADWIN drift detector
        adwin.add_element(test_acc)

        # Check for drift compared to previous runs
        if previous_acc is not None and adwin.detected_change():
            print(f"⚠️ Drift detected in Run {run+1}! Consider retraining the model.")

        previous_acc = test_acc  # Update previous accuracy

# ==========================
# Analyze MLflow Runs
# ==========================

# Initialize MLflow client
client = MlflowClient()

# Get Experiment ID
experiment_name = "FashionMNIST-Tracking"
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Fetch all runs from the experiment
runs = client.search_runs(experiment_id, order_by=["metrics.test_accuracy DESC"])

# Convert to Pandas DataFrame for analysis
df = pd.DataFrame([{  
    "run_id": run.info.run_id,  
    "test_accuracy": run.data.metrics["test_accuracy"],  
    "test_loss": run.data.metrics["test_loss"],  
    "duration": run.info.end_time - run.info.start_time,  
    "start_time": run.info.start_time
} for run in runs])

# Convert start_time to datetime format for better visualization
df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")

# Display sorted runs (best accuracy first)
print(df)

# ==========================
# Detect Data Drift
# ==========================

# Define threshold for significant accuracy drop (e.g., 5% or 0.05)
accuracy_drop_threshold = 0.05  

# Compute accuracy change between consecutive runs
df["accuracy_change"] = df["test_accuracy"].diff()

# Identify runs where accuracy drops beyond the threshold
drift_runs = df[df["accuracy_change"] < -accuracy_drop_threshold]

if not drift_runs.empty:
    print("⚠️ Possible data drift detected in the following runs:")
    print(drift_runs)
else:
    print("✅ No significant accuracy drops detected.")

# ==========================
# Plot Accuracy Over Time
# ==========================

plt.figure(figsize=(10, 5))
plt.plot(df["start_time"], df["test_accuracy"], marker="o", linestyle="-", label="Test Accuracy")
plt.axhline(y=df["test_accuracy"].max() - accuracy_drop_threshold, color="red", linestyle="--", label="Drift Threshold")
plt.xlabel("Run Timestamp")
plt.ylabel("Test Accuracy")
plt.title("Accuracy Over Time - Drift Detection")
plt.legend()
plt.xticks(rotation=45)
plt.savefig("models/mlflow_drift_detection.png")
print("✅ Drift detection plot saved as models/mlflow_drift_detection.png")
