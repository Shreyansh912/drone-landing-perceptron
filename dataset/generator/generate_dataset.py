import os
import numpy as np
import pandas as pd


def generate_landing_dataset(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    """
    Generates a 6-dimensional dataset representing candidate landing zones.
    Labels: 1 = SAFE, 0 = UNSAFE
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2

    # --- 1. SAFE SAMPLES (n_per_class) ---
    # Safe pads: large area, very low obstacles, far clearance, flat, moderate offset, high circularity
    area_safe = np.random.uniform(0.55, 0.95, n_per_class)
    obs_density_safe = np.random.uniform(0.01, 0.15, n_per_class)
    min_dist_safe = np.random.uniform(0.50, 0.98, n_per_class)
    flatness_safe = np.random.uniform(0.70, 0.99, n_per_class)
    offset_safe = np.random.uniform(0.02, 0.40, n_per_class)
    circularity_safe = np.random.uniform(0.65, 0.98, n_per_class)
    labels_safe = np.ones(n_per_class, dtype=int)

    # --- 2. UNSAFE SAMPLES (n_per_class) ---
    # Unsafe pads: small area OR high obstacle density OR poor clearance OR rough terrain
    area_unsafe = np.random.uniform(0.05, 0.50, n_per_class)
    obs_density_unsafe = np.random.uniform(0.30, 0.90, n_per_class)
    min_dist_unsafe = np.random.uniform(0.02, 0.40, n_per_class)
    flatness_unsafe = np.random.uniform(0.10, 0.60, n_per_class)
    offset_unsafe = np.random.uniform(0.35, 0.95, n_per_class)
    circularity_unsafe = np.random.uniform(0.10, 0.55, n_per_class)
    labels_unsafe = np.zeros(n_per_class, dtype=int)

    # Combine
    X = np.vstack([
        np.column_stack([area_safe, obs_density_safe, min_dist_safe, flatness_safe, offset_safe, circularity_safe]),
        np.column_stack([area_unsafe, obs_density_unsafe, min_dist_unsafe, flatness_unsafe, offset_unsafe, circularity_unsafe])
    ])
    y = np.concatenate([labels_safe, labels_unsafe])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    df = pd.DataFrame(X, columns=[
        "normalized_area",
        "obstacle_density",
        "min_obstacle_distance",
        "surface_flatness",
        "centroid_offset",
        "circularity"
    ])
    df["label"] = y
    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "landing_features.csv")

    df = generate_landing_dataset(n_samples=1200)
    df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Dataset generated with {len(df)} samples.")
    print(f"[PATH] Saved to: {os.path.abspath(out_path)}")
    print(df.head())