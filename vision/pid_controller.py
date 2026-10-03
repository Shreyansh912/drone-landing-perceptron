import numpy as np


class PIDController:
    """
    Discrete PID Controller with anti-windup clamping and derivative filtering.
    """
    def __init__(self, kp: float, ki: float, kd: float, max_output: float = 1.5, max_integral: float = 0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.max_integral = max_integral

        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup saturation
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -self.max_integral, self.max_integral))
        i_term = self.ki * self.integral

        # Derivative term
        if self.first_run:
            d_term = 0.0
            self.first_run = False
        else:
            d_term = self.kd * (error - self.prev_error) / dt

        self.prev_error = error

        # Calculate bounded velocity output
        output = p_term + i_term + d_term
        return float(np.clip(output, -self.max_output, self.max_output))