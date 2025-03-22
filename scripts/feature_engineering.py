import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import shap
import matplotlib.pyplot as plt

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("processed_data", exist_ok=True)

# Load Fashion MNIST
def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, y_train, X_test, y_test

# Feature Engineering: Scaling & PCA
def preprocess_data(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Save processed data
    np.save("processed_data/X_train_pca.npy", X_train_pca)
    np.save("processed_data/X_test_pca.npy", X_test_pca)
    np.save("processed_data/y_train.npy", y_train)  # ✅ Now properly passed
    np.save("processed_data/y_test.npy", y_test)  # ✅ Now properly passed

    return X_train_pca, X_test_pca

# Train a simple model
def train_model(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)
    
    # Save trained model
    model.save("models/fashion_mnist_model.h5")

    return model

# SHAP Explainability
def explainability_analysis(model, X_train):
    explainer = shap.KernelExplainer(lambda X: model.predict(X), X_train[:100])
    shap_values = explainer.shap_values(X_train[:100])
    shap.summary_plot(shap_values, X_train[:100])

# Main function
def main():
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    
    print("Preprocessing data...")
    X_train_pca, X_test_pca = preprocess_data(X_train, X_test, y_train, y_test)  # ✅ Pass y_train, y_test

    print("Training model...")
    model = train_model(X_train_pca, y_train)
    
    print("Running explainability analysis...")
    explainability_analysis(model, X_train_pca)

if __name__ == "__main__":
    main()
