import os
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from river.drift import ADWIN  
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
import mlflow.models.signature
from mlflow.types import Schema, TensorSpec

# ==========================
# Load & Preprocess Data
# ==========================

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, y_train = x_train[:10000], y_train[:10000]  
x_test, y_test = x_test[:2000], y_test[:2000]  

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ==========================
# MLflow Experiment Setup
# ==========================

mlflow.set_tracking_uri("./mlruns")  
mlflow.set_experiment("FashionMNIST-Tracking")

# Initialize ADWIN for drift detection
adwin = ADWIN()
previous_acc = None

# ==========================
# Train & Log Model in MLflow
# ==========================
for run in range(5):  
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

        # ✅ Define model input signature to fix MLflow warning
        input_schema = Schema([
            TensorSpec(np.dtype(np.float32), (-1, 28, 28, 1))
        ])
        mlflow.keras.log_model(model, "model", signature=mlflow.models.signature.ModelSignature(input_schema))

        print(f"Run {run+1}: Accuracy = {test_acc:.4f}, Loss = {test_loss:.4f}")

        # ✅ Fix ADWIN update method
        adwin.update(test_acc)  

        if previous_acc is not None and adwin.drift_detected:
            print(f"⚠️ Drift detected in Run {run+1}! Consider retraining the model.")

        previous_acc = test_acc  

# ==========================
# Analyze MLflow Runs
# ==========================

client = MlflowClient()
experiment_name = "FashionMNIST-Tracking"
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

runs = client.search_runs(experiment_id, order_by=["metrics.test_accuracy DESC"])

df = pd.DataFrame([{  
    "run_id": run.info.run_id,  
    "test_accuracy": run.data.metrics["test_accuracy"],  
    "test_loss": run.data.metrics["test_loss"],  
    "duration": run.info.end_time - run.info.start_time,  
    "start_time": run.info.start_time
} for run in runs])

df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")

print(df)

# ==========================
# Compare Runs by Accuracy
# ==========================

df_sorted = df.sort_values("test_accuracy", ascending=False)

plt.figure(figsize=(10, 5))
plt.barh(df_sorted["run_id"], df_sorted["test_accuracy"], color="blue")
plt.xlabel("Test Accuracy")
plt.ylabel("Run ID")
plt.title("Comparison of MLflow Runs by Test Accuracy")
plt.gca().invert_yaxis()  

# ✅ Ensure the models/ directory exists
os.makedirs("models", exist_ok=True)

# ✅ Save accuracy comparison plot
plt.savefig("models/mlflow_accuracy_comparison.png")
print("✅ Accuracy comparison plot saved as models/mlflow_accuracy_comparison.png")

# ==========================
# Detect Data Drift
# ==========================

accuracy_drop_threshold = 0.05  
df["accuracy_change"] = df["test_accuracy"].diff()
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

# ✅ Save drift detection plot
plt.savefig("models/mlflow_drift_detection.png")
print("✅ Drift detection plot saved as models/mlflow_drift_detection.png")
