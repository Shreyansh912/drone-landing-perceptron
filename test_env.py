import numpy as np
import cv2
import pandas as pd
import matplotlib

print("=" * 55)
print("  STAGE 1: SYSTEM & MATHEMATICAL VERIFICATION CHECK")
print("=" * 55)
print(f"[*] NumPy:      {np.__version__}")
print(f"[*] OpenCV:     {cv2.__version__}")
print(f"[*] Pandas:     {pd.__version__}")
print(f"[*] Matplotlib: {matplotlib.__version__}")
print("-" * 55)

# Verification vector math: z = w^T * x + b
x = np.array([2.0, 0.1, 0.8])
w = np.array([1.5, -8.0, 3.0])
b = -0.5
z = float(np.dot(w, x) + b)
y_hat = 1 if z >= 0 else 0

print(f"[*] Activation Score (z):  {z:.4f}")
print(f"[*] Perceptron Prediction: {'SAFE (1)' if y_hat == 1 else 'UNSAFE (0)'}")
print("-" * 55)
assert y_hat == 1, "Verification error."
print("[SUCCESS] All libraries imported and vector math validated!")
print("=" * 55)