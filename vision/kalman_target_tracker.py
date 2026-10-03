import numpy as np


class TargetKalmanFilter:
    """4-State Discrete Linear Kalman Filter for Moving Target Position & Velocity Estimation."""
    def __init__(self, dt: float = 0.1, process_noise_std: float = 0.08, meas_noise_std: float = 0.03):
        self.dt = dt

        # State transition matrix F
        self.F = np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Measurement matrix H (observing position x, y)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Process noise covariance Q
        q_pos = (process_noise_std**2) * (0.5 * dt**2)
        q_vel = (process_noise_std**2) * dt
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])

        # Measurement noise covariance R
        self.R = np.eye(2, dtype=np.float64) * (meas_noise_std**2)

        # State estimate x and covariance P
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1.0
        self.initialized = False

    def initialize(self, init_x: float, init_y: float):
        self.x = np.array([[init_x], [init_y], [0.0], [0.0]], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 0.1
        self.initialized = True

    def predict(self):
        """A priori state and covariance propagation."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.flatten()

    def update(self, meas_x: float, meas_y: float):
        """Measurement update / Kalman gain correction."""
        if not self.initialized:
            self.initialize(meas_x, meas_y)
            return self.x.flatten()

        z = np.array([[meas_x], [meas_y]], dtype=np.float64)
        y = z - self.H @ self.x  # Innovation / residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return self.x.flatten()

    def get_state(self):
        return self.x.flatten()