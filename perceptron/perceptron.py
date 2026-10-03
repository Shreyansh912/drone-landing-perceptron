import numpy as np


class Perceptron:
    """
    Pure NumPy implementation of Frank Rosenblatt's Perceptron classifier.
    
    Parameters:
    -----------
    learning_rate : float
        Step size parameter eta in (0.0, 1.0].
    n_epochs : int
        Maximum passes over the training dataset.
    random_state : int or None
        Seed for reproducible weight initialization.
    """
    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 50, random_state: int = 42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Perceptron model to training features and binary targets.

        Parameters:
        -----------
        X : np.ndarray of shape (N, D)
            Feature matrix with N samples and D dimensions.
        y : np.ndarray of shape (N,)
            Ground truth binary target vector (values must be 0 or 1).

        Returns:
        --------
        self : Perceptron
        """
        # Ensure inputs are valid numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)

        n_samples, n_features = X.shape

        # Initialize weights with small random Gaussian values to break symmetry
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0.0
        self.errors_per_epoch = []

        # Training loop
        for epoch in range(1, self.n_epochs + 1):
            misclassifications = 0
            
            for xi, target in zip(X, y):
                # 1. Compute prediction
                prediction = self.predict_single(xi)
                
                # 2. Compute error: e in {-1, 0, 1}
                error = target - prediction
                
                # 3. Update parameters if there was an error
                if error != 0:
                    update = self.learning_rate * error
                    self.weights += update * xi
                    self.bias += update
                    misclassifications += 1

            self.errors_per_epoch.append(misclassifications)

            # Early stopping: if all points are correctly classified, convergence is reached
            if misclassifications == 0:
                print(f"[CONVERGED] Perfect linear separation reached at Epoch {epoch}/{self.n_epochs}.")
                break

        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculate linear activation z = w^T * x + b."""
        return np.dot(X, self.weights) + self.bias

    def predict_single(self, x: np.ndarray) -> int:
        """Predict binary class for a single feature vector."""
        z = np.dot(x, self.weights) + self.bias
        return 1 if z >= 0.0 else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels for an array of feature vectors.
        
        Parameters:
        -----------
        X : np.ndarray of shape (N, D)
        
        Returns:
        --------
        np.ndarray of shape (N,) with values in {0, 1}
        """
        X = np.asarray(X, dtype=np.float64)
        z = self.net_input(X)
        return np.where(z >= 0.0, 1, 0)