import os
import cv2
import numpy as np


class LandingZoneEvaluator:
    """
    Evaluates candidate landing zones using the trained custom Perceptron.
    Loads learned weights and scaler parameters from disk.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "..", "perceptron", "trained_model.npz")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Run perceptron/train.py first.")

        # Load weights, bias, and Z-score scaler parameters
        data = np.load(model_path, allow_pickle=True)
        self.weights = data["weights"]
        self.bias = float(data["bias"])
        self.scaler_mean = data["scaler_mean"]
        self.scaler_scale = data["scaler_scale"]
        self.feature_names = data["features"]

        print(f"[EVALUATOR] Loaded trained Perceptron model. Weights: {self.weights}, Bias: {self.bias:.4f}")

    def evaluate(self, raw_features: np.ndarray):
        """
        Takes raw 6D features, standardizes them, and runs Perceptron inference.
        
        Returns:
        --------
        is_safe : bool (True = SAFE, False = UNSAFE)
        score : float (Net activation z)
        """
        # 1. Z-Score Standardization: (x - mu) / sigma
        scaled_features = (raw_features - self.scaler_mean) / self.scaler_scale

        # 2. Linear projection: z = w^T * x' + b
        z = float(np.dot(scaled_features, self.weights) + self.bias)

        # 3. Decision boundary check
        is_safe = (z >= 0.0)

        return is_safe, z