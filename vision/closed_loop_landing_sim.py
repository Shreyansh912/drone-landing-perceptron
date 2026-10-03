import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessor import ImagePreprocessor
from landing_detector import LandingZoneDetector
from landing_evaluator import LandingZoneEvaluator
from pid_controller import PIDController


class DroneLandingSimulator:
    def __init__(self):
        # 1. Perception and decision modules
        self.preprocessor = ImagePreprocessor()
        self.detector = LandingZoneDetector(min_area=150.0, max_area=500000.0)
        self.evaluator = LandingZoneEvaluator()

        # 2. 3-Axis PID Controllers
        self.pid_x = PIDController(kp=1.15, ki=0.04, kd=0.38, max_output=1.4)
        self.pid_y = PIDController(kp=1.15, ki=0.04, kd=0.38, max_output=1.4)
        self.pid_z = PIDController(kp=0.55, ki=0.02, kd=0.18, max_output=0.4)

        # 3. Physical Drone State (Meters): Starts at (-1.2m, +0.8m, 4.5m)
        self.drone_pos = np.array([-1.2, 0.8, 4.5], dtype=np.float64)
        self.pad_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # Safe target pad at origin

        # Telemetry history records
        self.history = {
            "time": [], "x": [], "y": [], "z": [],
            "error_x": [], "error_y": [], "score_z": [], "state": []
        }

    def render_synthetic_camera(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Projects simulated 3D world elements onto the downward 2D camera plane."""
        frame = np.full((height, width, 3), 40, dtype=np.uint8)  # Ground terrain

        # Scale factor: pixels per ground meter
        alt = max(self.drone_pos[2], 0.20)
        px_per_m = 360.0 / alt
        u0, v0 = width / 2.0, height / 2.0

        # Safe Landing Pad at (0.0, 0.0) - Clean, uniform concrete pad
        pad_px_x = int(u0 + (self.pad_pos[0] - self.drone_pos[0]) * px_per_m)
        pad_px_y = int(v0 + (self.pad_pos[1] - self.drone_pos[1]) * px_per_m)
        pad_radius = int(0.65 * px_per_m)

        if pad_radius > 4:
            # Flat, high-contrast, obstacle-free landing zone
            cv2.circle(frame, (pad_px_x, pad_px_y), pad_radius, (190, 190, 190), -1)
            cv2.circle(frame, (pad_px_x, pad_px_y), pad_radius, (230, 230, 230), 2)

        # Unsafe Debris/Obstacle Field at (-2.0, -0.7)
        obs_px_x = int(u0 + (-2.0 - self.drone_pos[0]) * px_per_m)
        obs_px_y = int(v0 + (-0.7 - self.drone_pos[1]) * px_per_m)
        obs_sz = int(0.55 * px_per_m)
        if obs_sz > 4:
            cv2.rectangle(frame, (obs_px_x - obs_sz, obs_px_y - obs_sz), (obs_px_x + obs_sz, obs_px_y + obs_sz), (110, 110, 110), -1)
            for dx in [-obs_sz//2, 0, obs_sz//2]:
                for dy in [-obs_sz//2, 0, obs_sz//2]:
                    cv2.circle(frame, (obs_px_x + dx, obs_px_y + dy), max(2, obs_sz // 5), (10, 10, 10), -1)

        return frame

    def extract_features_altitude_aware(self, contour, gray_frame, edge_map, width=640, height=480):
        """Extracts scale-invariant 6D features calibrated to current altitude."""
        M = cv2.moments(contour)
        if M["m00"] <= 1e-4:
            return None, None

        area = float(M["m00"])
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Scale-invariant area ratio
        alt = max(self.drone_pos[2], 0.20)
        px_per_m = 360.0 / alt
        expected_pad_area = np.pi * (0.65 * px_per_m)**2
        f1_area = float(np.clip(0.80 * (area / max(expected_pad_area, 1.0)), 0.10, 0.95))

        # Obstacle Density inside contour
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        candidate_edges = cv2.bitwise_and(edge_map, edge_map, mask=mask)
        f2_obstacle_density = float(np.count_nonzero(candidate_edges)) / max(area, 1.0)
        f2_obstacle_density = min(1.0, f2_obstacle_density)

        # Clearance to nearest obstacle outside the pad
        inv_edges = cv2.bitwise_not(edge_map)
        dist_transform = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 5)
        cx_int = int(np.clip(cx, 0, width - 1))
        cy_int = int(np.clip(cy, 0, height - 1))
        # Distance from pad perimeter to nearest obstacle
        f3_min_dist = float(np.clip(dist_transform[cy_int, cx_int] / (px_per_m * 0.8), 0.10, 0.95))

        # Surface Flatness (clean uniform pad -> high flatness)
        _, std_val = cv2.meanStdDev(gray_frame, mask=mask)
        f4_flatness = 1.0 - min(1.0, float(std_val[0][0]) / 40.0)

        # Centroid offset from camera optical center
        dx = cx - width / 2.0
        dy = cy - height / 2.0
        max_offset = np.sqrt((width / 2.0)**2 + (height / 2.0)**2)
        f5_offset = min(1.0, np.sqrt(dx**2 + dy**2) / max_offset)

        # Circularity (isoperimetric quotient)
        perimeter = cv2.arcLength(contour, closed=True)
        f6_circularity = (4.0 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
        f6_circularity = min(1.0, f6_circularity)

        features = np.array([f1_area, f2_obstacle_density, f3_min_dist, f4_flatness, f5_offset, f6_circularity])
        spatial = {"centroid": (cx, cy), "error_x": dx, "error_y": dy, "area": area}
        return features, spatial

    def run_simulation(self, total_time: float = 20.0, dt: float = 0.1):
        print("=" * 65)
        print("STAGE 6: CLOSED-LOOP AUTONOMOUS LANDING SIMULATION")
        print("=" * 65)

        state = "SEARCH"
        time_elapsed = 0.0
        aligned_counter = 0

        while time_elapsed <= total_time:
            # 1. Perception frame capture
            frame = self.render_synthetic_camera()
            _, gray, edges = self.preprocessor.process(frame)
            candidates, _ = self.detector.detect_candidates(gray)

            best_target = None
            max_score = -1e9

            # 2. Extract features and evaluate via Perceptron
            for c in candidates:
                features, spatial = self.extract_features_altitude_aware(c, gray, edges)
                if features is None:
                    continue

                is_safe, z_score = self.evaluator.evaluate(features)

                # Linear decision boundary: SAFE if z >= 0.0
                if is_safe:
                    utility = z_score - 0.20 * features[4]
                    if utility > max_score:
                        max_score = utility
                        best_target = spatial

            # 3. Closed-Loop State Machine
            ex_meters, ey_meters = 0.0, 0.0
            vz_cmd = 0.0

            if best_target is not None:
                alt = max(self.drone_pos[2], 0.20)
                px_per_m = 360.0 / alt
                ex_meters = best_target["error_x"] / px_per_m
                ey_meters = best_target["error_y"] / px_per_m
                horizontal_error = np.sqrt(ex_meters**2 + ey_meters**2)

                if state in ["SEARCH", "ABORT"]:
                    state = "ALIGN"

                if state == "ALIGN":
                    vz_cmd = 0.0  # Hold search altitude while centering
                    if horizontal_error < 0.15:  # Centered within 15 cm
                        aligned_counter += 1
                        if aligned_counter >= 3:
                            state = "DESCEND"
                    else:
                        aligned_counter = 0

                elif state == "DESCEND":
                    vz_cmd = -0.32  # Steady descent at 32 cm/s
                    if horizontal_error > 0.40:
                        state = "ALIGN"
                        vz_cmd = 0.0
                        aligned_counter = 0
                    elif self.drone_pos[2] <= 0.18:
                        state = "TOUCHDOWN"
                        vz_cmd = 0.0
            else:
                state = "ABORT"
                vz_cmd = 0.05
                aligned_counter = 0

            # 4. Compute PID Commands
            vx_cmd = self.pid_x.compute(ex_meters, dt) if state != "TOUCHDOWN" else 0.0
            vy_cmd = self.pid_y.compute(ey_meters, dt) if state != "TOUCHDOWN" else 0.0

            # 5. Integrate Kinematics
            if state != "TOUCHDOWN":
                self.drone_pos[0] += vx_cmd * dt
                self.drone_pos[1] += vy_cmd * dt
                self.drone_pos[2] += vz_cmd * dt
                self.drone_pos[2] = max(0.0, self.drone_pos[2])

            # 6. Record Telemetry
            self.history["time"].append(time_elapsed)
            self.history["x"].append(self.drone_pos[0])
            self.history["y"].append(self.drone_pos[1])
            self.history["z"].append(self.drone_pos[2])
            self.history["error_x"].append(ex_meters)
            self.history["error_y"].append(ey_meters)
            self.history["score_z"].append(max_score if best_target else -1.0)
            self.history["state"].append(state)

            if int(time_elapsed * 10) % 15 == 0:
                h_err = np.sqrt(ex_meters**2 + ey_meters**2)
                print(
                    f"T={time_elapsed:4.1f}s | State={state:<9} | Alt={self.drone_pos[2]:.2f}m | "
                    f"Pos=({self.drone_pos[0]:+.2f}, {self.drone_pos[1]:+.2f})m | HorizErr={h_err:.2f}m"
                )

            if state == "TOUCHDOWN":
                print("\n" + "=" * 65)
                print(f"[SUCCESS] Touchdown achieved at T = {time_elapsed:.1f}s!")
                final_err = np.sqrt(self.drone_pos[0]**2 + self.drone_pos[1]**2)
                print(f"Final Landing Position Error: {final_err * 100:.2f} cm from pad center.")
                print("=" * 65)
                break

            time_elapsed += dt

        self.plot_telemetry()

    def plot_telemetry(self):
        """Generates and saves the 3-tier flight trajectory telemetry plot."""
        out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(out_dir, exist_ok=True)
        out_plot = os.path.join(out_dir, "stage6_landing_trajectory.png")

        t = self.history["time"]
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # 1. Altitude
        axs[0].plot(t, self.history["z"], color="royalblue", linewidth=2.2, label="Altitude Z [m]")
        axs[0].axhline(0.18, color="red", linestyle="--", label="Touchdown Threshold (0.18m)")
        axs[0].set_ylabel("Altitude [m]")
        axs[0].grid(True, linestyle="--", alpha=0.6)
        axs[0].legend(loc="upper right")
        axs[0].set_title("Autonomous Drone Landing Trajectory & State Transitions")

        # 2. Horizontal Position
        axs[1].plot(t, self.history["x"], label="Drone X [m]", color="crimson")
        axs[1].plot(t, self.history["y"], label="Drone Y [m]", color="forestgreen")
        axs[1].axhline(0.0, color="black", linestyle=":", label="Landing Pad Center (0,0)")
        axs[1].set_ylabel("Horizontal Pos [m]")
        axs[1].grid(True, linestyle="--", alpha=0.6)
        axs[1].legend(loc="upper right")

        # 3. State Sequence
        state_map = {"SEARCH": 0, "ALIGN": 1, "DESCEND": 2, "ABORT": 3, "TOUCHDOWN": 4}
        state_vals = [state_map.get(s, 0) for s in self.history["state"]]
        axs[2].step(t, state_vals, color="purple", linewidth=2, where="post")
        axs[2].set_yticks(list(state_map.values()))
        axs[2].set_yticklabels(list(state_map.keys()))
        axs[2].set_ylabel("FSM State")
        axs[2].set_xlabel("Time [s]")
        axs[2].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(out_plot, dpi=200)
        print(f"\n[PLOT SAVED] Telemetry plot saved to: {os.path.abspath(out_plot)}")
        plt.show()


if __name__ == "__main__":
    sim = DroneLandingSimulator()
    sim.run_simulation()