import os
import time
import numpy as np
import pandas as pd
from perceptron import Perceptron


class LogisticRegressionSGD:
    """Logistic Regression trained via Stochastic Gradient Descent."""
    def __init__(self, lr=0.05, n_epochs=50):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for _ in range(self.n_epochs):
            for xi, target in zip(X, y):
                z = np.dot(xi, self.weights) + self.bias
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))
                err = target - p
                self.weights += self.lr * err * xi
                self.bias += self.lr * err
        return self

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))
        return np.where(p >= 0.5, 1, 0)


class SmallMLP:
    """2-Layer Perceptron (6 -> 8 -> 1) with ReLU and Sigmoid."""
    def __init__(self, lr=0.05, n_epochs=50, hidden_dim=8):
        self.lr = lr
        self.n_epochs = n_epochs
        self.hidden_dim = hidden_dim
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = 0.0

    def fit(self, X, y):
        np.random.seed(42)
        n_samples, n_features = X.shape
        self.W1 = np.random.randn(n_features, self.hidden_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim) * 0.1
        self.b2 = 0.0

        for _ in range(self.n_epochs):
            for xi, target in zip(X, y):
                # Forward pass
                h = np.maximum(0, np.dot(xi, self.W1) + self.b1)  # ReLU
                z = np.dot(h, self.W2) + self.b2
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

                # Backward pass
                dz = p - target
                dW2 = dz * h
                db2 = dz
                dh = dz * self.W2 * (h > 0)
                dW1 = np.outer(xi, dh)
                db1 = dh

                # Update
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
        return self

    def predict(self, X):
        h = np.maximum(0, np.dot(X, self.W1) + self.b1)
        z = np.dot(h, self.W2) + self.b2
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))
        return np.where(p >= 0.5, 1, 0)


def run_benchmark():
    print("=" * 70)
    print("STAGE 9: CLASSIFIER BASELINE BENCHMARK")
    print("=" * 70)

    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "landing_features.csv")
    df = pd.read_csv(dataset_path)

    X = df[[c for c in df.columns if c != "label"]].values
    y = df["label"].values

    # Standardize
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    X_scaled = (X - mean) / scale

    # Train/Test Split (80/20)
    n_train = int(len(X) * 0.8)
    X_train, y_train = X_scaled[:n_train], y[:n_train]
    X_test, y_test = X_scaled[n_train:], y[n_train:]

    models = {
        "Custom Perceptron": Perceptron(learning_rate=0.01, n_epochs=50),
        "Logistic Regression": LogisticRegressionSGD(lr=0.05, n_epochs=50),
        "Small MLP (6-8-1)": SmallMLP(lr=0.05, n_epochs=50, hidden_dim=8)
    }

    comparison = []

    for name, clf in models.items():
        clf.fit(X_train, y_train)

        # Latency Benchmark: 10,000 inferences
        start_time = time.perf_counter()
        for _ in range(10000):
            _ = clf.predict(X_test[:1])
        elapsed = time.perf_counter() - start_time
        latency_us = (elapsed / 10000.0) * 1e6

        preds = clf.predict(X_test)

        tp = np.sum((y_test == 1) & (preds == 1))
        fp = np.sum((y_test == 0) & (preds == 1))
        tn = np.sum((y_test == 0) & (preds == 0))
        fn = np.sum((y_test == 1) & (preds == 0))

        acc = (tp + tn) / len(y_test)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fsr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        param_count = 7 if name != "Small MLP (6-8-1)" else (6 * 8 + 8 + 8 + 1)

        comparison.append({
            "Model": name,
            "Accuracy": f"{acc * 100:.1f}%",
            "F1-Score": f"{f1:.4f}",
            "False-Safe Rate": f"{fsr * 100:.2f}%",
            "Latency (μs)": f"{latency_us:.2f} μs",
            "Parameters": param_count,
            "Interpretability": "High (Direct Weights)" if name != "Small MLP (6-8-1)" else "Low (Non-linear)"
        })

    comp_df = pd.DataFrame(comparison)
    print("\n" + comp_df.to_string(index=False))

    out_csv = os.path.join(os.path.dirname(__file__), "..", "results", "model_comparison.csv")
    comp_df.to_csv(out_csv, index=False)
    print(f"\n[BENCHMARK EXPORTED] Benchmark saved to: {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    run_benchmark()