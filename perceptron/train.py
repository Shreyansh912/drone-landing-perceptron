import os
import numpy as np
import pandas as pd
from perceptron import Perceptron
from metrics import classification_report, print_report


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0  # prevent division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def split_data(X: np.ndarray, y: np.ndarray, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def run_pipeline():
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "landing_features.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing {dataset_path}. Run generate_dataset.py first.")

    # Load dataset
    df = pd.read_csv(dataset_path)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    # Train / Val / Test Split (70% / 15% / 15%)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train custom Perceptron
    clf = Perceptron(learning_rate=0.01, n_epochs=50, random_state=42)
    clf.fit(X_train_scaled, y_train)

    print("\nLearned Weights for 6D Landing Perception:")
    for col, w in zip(feature_cols, clf.weights):
        print(f"  {col:<24}: {w:+.4f}")
    print(f"  {'bias':<24}: {clf.bias:+.4f}\n")

    # Evaluate on Unseen Test Set
    test_preds = clf.predict(X_test_scaled)
    metrics = classification_report(y_test, test_preds)
    print_report(metrics)

    # Save learned parameters for the ROS 2 node
    np.savez(
        os.path.join(os.path.dirname(__file__), "trained_model.npz"),
        weights=clf.weights,
        bias=clf.bias,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        features=feature_cols
    )
    print("[MODEL SAVED] Weights and scaler saved to trained_model.npz")


if __name__ == "__main__":
    run_pipeline()