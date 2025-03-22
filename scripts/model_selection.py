import numpy as np
import pandas as pd
import optuna
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import fashion_mnist

# =======================
# 📌 Load & Preprocess Data
# =======================

# Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Reduce dataset size for faster training (1,000 train samples, 300 test samples)
X_train, X_test, y_train, y_test = X_train[:1000], X_test[:300], y_train[:1000], y_test[:300]  

# Flatten images (convert 28x28 pixels to a 1D vector) & Normalize (scale pixel values between 0-1)
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# Split train set further into train (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# =======================
# 📌 Feature Scaling
# =======================

# Standardize the dataset (mean=0, variance=1) to improve model convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# =======================
# 📌 Baseline Model Training (XGBoost)
# =======================

# Train an initial XGBoost model with default hyperparameters
model = xgb.XGBClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
model.fit(X_train, y_train)

# Evaluate the baseline model
baseline_accuracy = accuracy_score(y_val, model.predict(X_val))
print("Baseline Model Accuracy:", baseline_accuracy)

# =======================
# 📌 Hyperparameter Optimization with Optuna
# =======================

def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.

    Args:
    trial: Optuna trial object to suggest hyperparameters.

    Returns:
    accuracy_score: Validation accuracy score for the model with suggested hyperparameters.
    """
    # Suggest hyperparameters for tuning
    n_estimators = trial.suggest_int("n_estimators", 50, 150, step=50)  # Number of trees
    max_depth = trial.suggest_int("max_depth", 3, 12, step=3)  # Depth of each tree
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, step=0.05)  # Learning rate

    # Train an XGBoost model with suggested hyperparameters
    model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, 
                              random_state=42, n_jobs=1)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

# Run Optuna to optimize hyperparameters over 5 trials
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)

# =======================
# 📌 Train Final Model with Optimized Hyperparameters
# =======================

# Get the best hyperparameters from Optuna study
best_params = study.best_params

# Train the final model using the best hyperparameters
final_model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=1)
final_model.fit(X_train, y_train)

# Predict on the test set
y_pred = final_model.predict(X_test)

# Evaluate the final model
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy: {final_accuracy:.4f}")
print("Best Hyperparameters:", best_params)

# =======================
# 📌 Explainability with SHAP
# =======================

# Create a SHAP explainer for feature importance
explainer = shap.Explainer(final_model, X_train)

# Compute SHAP values for a small subset of the test data
shap_values = explainer(X_test[:50])

# Generate a SHAP summary plot to visualize feature importance
shap.summary_plot(shap_values, X_test[:50])
