import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import sweetviz as sv
import os
from sweetviz.feature_config import FeatureConfig

def load_fashion_mnist():
    """Load Fashion MNIST dataset as DataFrame."""
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Flatten images
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Convert to DataFrame
    columns = [f'pixel_{i}' for i in range(X_train_flat.shape[1])]
    df_train = pd.DataFrame(X_train_flat, columns=columns)
    df_train['label'] = y_train.astype(int)

    df_test = pd.DataFrame(X_test_flat, columns=columns)
    df_test['label'] = y_test.astype(int)

    return df_train, df_test

def find_key_insights(df):
    """Extract key insights from the dataset."""
    insights = {}

    # 1️⃣ Check Class Distribution
    class_counts = df['label'].value_counts(normalize=True) * 100
    imbalance = class_counts.max() - class_counts.min()
    insights['class_distribution'] = f"Class imbalance: {imbalance:.2f}%"

    # 2️⃣ Find Missing Values
    missing_values = df.isnull().sum().sum()
    insights['missing_values'] = f"Total missing values: {missing_values}"

    # 3️⃣ Detect Low-Variance Features
    feature_variances = df.drop(columns=['label']).var()
    low_variance_features = feature_variances[feature_variances < 1.0].index.tolist()
    insights['low_variance'] = f"Low variance features: {len(low_variance_features)}"

    # 4️⃣ Identify Highly Correlated Features
    corr_matrix = df.drop(columns=['label']).corr().abs()
    high_corr_pairs = [(i, j, corr_matrix.loc[i, j]) for i in corr_matrix.columns for j in corr_matrix.columns if i != j and corr_matrix.loc[i, j] > 0.9]
    insights['high_correlation'] = f"Highly correlated features: {len(high_corr_pairs)}"

    return insights

def generate_eda_report(df, output_path, sample_size=5000):
    """Generate EDA report with a smaller sample for faster processing."""
    
    # ✅ Take a smaller subset for speed
    df_sample = df.sample(n=min(len(df), sample_size), random_state=42)
    
    # Ensure 'label' is numeric
    df_sample['label'] = df_sample['label'].astype(int)

    # Define feature configuration (force 'label' as numeric)
    feat_cfg = FeatureConfig(force_num=['label'])
    
    # Generate report
    report = sv.analyze(df_sample, target_feat='label', feat_cfg=feat_cfg, pairwise_analysis='off')
    report.show_html(output_path)
    print(f"✅ EDA report saved at: {output_path}")

def main():
    """Main function to generate EDA and extract insights."""
    df_train, df_test = load_fashion_mnist()

    output_dir = "eda_reports"
    os.makedirs(output_dir, exist_ok=True)

    train_report_path = os.path.join(output_dir, "fashion_mnist_train_eda.html")
    test_report_path = os.path.join(output_dir, "fashion_mnist_test_eda.html")

    print("Generating reports...")
    generate_eda_report(df_train, train_report_path)
    generate_eda_report(df_test, test_report_path)

    # Extract key insights
    insights = find_key_insights(df_train)
    print("\n🔎 Key Insights:")
    for key, value in insights.items():
        print(f"✅ {key}: {value}")

if __name__ == "__main__":
    main()