import numpy as np
from perceptron import Perceptron
from visualize import plot_decision_regions


def generate_synthetic_landing_data(n_samples: int = 100, seed: int = 42):
    """
    Generates synthetic 2D candidate landing-zone data.
    Feature 1: Normalized Area (Larger is safer)
    Feature 2: Obstacle Density (Lower is safer)
    """
    np.random.seed(seed)
    half = n_samples // 2
    
    # SAFE Zones: high area (0.6 to 0.95), low obstacle density (0.01 to 0.25)
    area_safe = np.random.uniform(0.60, 0.95, half)
    density_safe = np.random.uniform(0.02, 0.25, half)
    X_safe = np.column_stack((area_safe, density_safe))
    y_safe = np.ones(half, dtype=int)
    
    # UNSAFE Zones: small area (0.05 to 0.45) OR high obstacle density (0.45 to 0.95)
    area_unsafe = np.random.uniform(0.10, 0.50, half)
    density_unsafe = np.random.uniform(0.40, 0.90, half)
    X_unsafe = np.column_stack((area_unsafe, density_unsafe))
    y_unsafe = np.zeros(half, dtype=int)
    
    X = np.vstack((X_safe, X_unsafe))
    y = np.concatenate((y_safe, y_unsafe))
    
    # Shuffle dataset
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 2: TRAINING CUSTOM PERCEPTRON FROM FIRST PRINCIPLES")
    print("=" * 60)

    # 1. Generate linearly separable landing candidate data
    X, y = generate_synthetic_landing_data(n_samples=100)
    print(f"[*] Generated {len(X)} landing-zone candidates (2 features: Area, Obstacle Density).")

    # 2. Instantiate and train model
    model = Perceptron(learning_rate=0.05, n_epochs=30, random_state=42)
    model.fit(X, y)

    # 3. Print learned parameters
    print("-" * 60)
    print(f"[*] Learned Weight for Area (w1):             {model.weights[0]:+.4f}")
    print(f"[*] Learned Weight for Obstacle Density (w2): {model.weights[1]:+.4f}")
    print(f"[*] Learned Bias Term (b):                    {model.bias:+.4f}")
    print("-" * 60)

    # 4. Physical sanity validation:
    # A safe zone must have high area (positive weight) and low obstacle density (negative weight)
    assert model.weights[0] > 0, "Error: Area should have a positive weight!"
    assert model.weights[1] < 0, "Error: Obstacle density should have a negative weight!"
    print("[VALIDATION PASSED] Learned signs match physical drone safety requirements.")

    # 5. Evaluate on novel candidate zones
    test_candidates = np.array([
        [0.85, 0.05],  # Large open pad, almost no obstacles -> Expected SAFE (1)
        [0.20, 0.80]   # Small narrow spot, high obstacle debris -> Expected UNSAFE (0)
    ])
    preds = model.predict(test_candidates)
    scores = model.net_input(test_candidates)
    
    print("\nNovel Inference Tests:")
    for i, (cand, p, score) in enumerate(zip(test_candidates, preds, scores)):
        status = "SAFE (1)" if p == 1 else "UNSAFE (0)"
        print(f"  Candidate {i+1}: Area={cand[0]:.2f}, ObstacleDensity={cand[1]:.2f} => Score={score:+.3f} => {status}")

    # 6. Plot boundary
    print("\nDisplaying convergence curve and decision boundary...")
    plot_decision_regions(X, y, model, feature_names=["Normalized Area (x1)", "Obstacle Density (x2)"])