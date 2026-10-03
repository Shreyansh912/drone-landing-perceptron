import numpy as np
import matplotlib.pyplot as plt


def plot_decision_regions(X: np.ndarray, y: np.ndarray, classifier, feature_names=None):
    """
    Visualize 2D data points and the linear decision boundary w1*x1 + w2*x2 + b = 0.
    """
    if X.shape[1] != 2:
        raise ValueError("Decision boundary plotting is only supported for 2D feature spaces.")

    plt.figure(figsize=(10, 5))

    # --- Subplot 1: Convergence History ---
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(classifier.errors_per_epoch) + 1), classifier.errors_per_epoch, marker='o', color='crimson')
    plt.xlabel('Epochs')
    plt.ylabel('Misclassifications (Errors)')
    plt.title('Perceptron Convergence Curve')
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- Subplot 2: Decision Boundary in 2D Feature Space ---
    plt.subplot(1, 2, 2)
    
    # Scatter plot data points
    safe_idx = (y == 1)
    unsafe_idx = (y == 0)
    
    plt.scatter(X[safe_idx, 0], X[safe_idx, 1], color='dodgerblue', marker='o', label='SAFE (1)', edgecolors='k', s=60)
    plt.scatter(X[unsafe_idx, 0], X[unsafe_idx, 1], color='firebrick', marker='x', label='UNSAFE (0)', s=60)

    # Calculate line: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
    w1, w2 = classifier.weights[0], classifier.weights[1]
    b = classifier.bias
    
    x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x1_vals = np.linspace(x1_min, x1_max, 100)
    
    if abs(w2) > 1e-6:
        x2_vals = -(w1 * x1_vals + b) / w2
        plt.plot(x1_vals, x2_vals, color='darkgreen', linestyle='-', linewidth=2.5, label='Decision Boundary (z=0)')
    
    # Set labels
    xlabel = feature_names[0] if feature_names else "Feature 1"
    ylabel = feature_names[1] if feature_names else "Feature 2"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Learned Separation Boundary')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()