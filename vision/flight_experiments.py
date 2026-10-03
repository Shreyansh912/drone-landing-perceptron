import os
import cv2
import numpy as np
import pandas as pd
from preprocessor import ImagePreprocessor
from landing_detector import LandingZoneDetector
from landing_evaluator import LandingZoneEvaluator
from pid_controller import PIDController


class DroneExperimentHarness:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.detector = LandingZoneDetector(min_area=150.0, max_area=500000.0)
        self.evaluator = LandingZoneEvaluator()

    def run_trial(self, scenario_name: str, wind_disturbance: bool = False, dynamic_obstacle: bool = False):
        """
        Executes a deterministic simulated trial.
        
        Parameters:
        -----------
        scenario_name : str ('nominal', 'cluttered', 'windy', 'dynamic_abort')
        wind_disturbance : bool (Adds persistent horizontal drift force)
        dynamic_obstacle : bool (Spawns an obstacle on the pad midway through descent)
        """
        # Controllers
        pid_x = PIDController(kp=1.15, ki=0.04, kd=0.38, max_output=1.4)
        pid_y = PIDController(kp=1.15, ki=0.04, kd=0.38, max_output=1.4)

        # Initial drone state
        drone_pos = np.array([-1.3, 0.85, 4.5], dtype=np.float64)
        pad_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        time_elapsed = 0.0
        dt = 0.1
        state = "SEARCH"
        aligned_counter = 0

        # Metrics recording
        history_x = []
        history_y = []
        history_z = []
        aborted = False
        landed = False

        while time_elapsed <= 25.0:
            alt = max(drone_pos[2], 0.20)
            px_per_m = 360.0 / alt
            u0, v0 = 320.0, 240.0

            # 1. Render camera image
            frame = np.full((480, 640, 3), 40, dtype=np.uint8)

            # Check if dynamic obstacle appears midway through descent (below 2.5m)
            pad_is_corrupted = (scenario_name == 'cluttered') or (dynamic_obstacle and drone_pos[2] < 2.5)

            pad_px_x = int(u0 + (pad_pos[0] - drone_pos[0]) * px_per_m)
            pad_px_y = int(v0 + (pad_pos[1] - drone_pos[1]) * px_per_m)
            pad_radius = int(0.65 * px_per_m)

            if pad_radius > 4:
                cv2.circle(frame, (pad_px_x, pad_px_y), pad_radius, (190, 190, 190), -1)
                cv2.circle(frame, (pad_px_x, pad_px_y), pad_radius, (230, 230, 230), 2)

                if pad_is_corrupted:
                    # Draw clutter/debris across the pad
                    for dx in [-pad_radius//3, 0, pad_radius//3]:
                        for dy in [-pad_radius//3, 0, pad_radius//3]:
                            cv2.circle(frame, (pad_px_x + dx, pad_px_y + dy), max(2, pad_radius // 4), (10, 10, 10), -1)

            # 2. Perception
            _, gray, edges = self.preprocessor.process(frame)
            candidates, _ = self.detector.detect_candidates(gray)

            best_target = None
            max_score = -1e9

            for c in candidates:
                M = cv2.moments(c)
                if M["m00"] <= 1e-4:
                    continue
                area = float(M["m00"])
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

                expected_pad_area = np.pi * (0.65 * px_per_m)**2
                f1 = float(np.clip(0.80 * (area / max(expected_pad_area, 1.0)), 0.10, 0.95))

                mask = np.zeros((480, 640), dtype=np.uint8)
                cv2.drawContours(mask, [c], -1, 255, -1)
                cand_edges = cv2.bitwise_and(edges, edges, mask=mask)
                f2 = float(np.count_nonzero(cand_edges)) / max(area, 1.0)

                inv_edges = cv2.bitwise_not(edges)
                dist_tf = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 5)
                cx_int, cy_int = int(np.clip(cx, 0, 639)), int(np.clip(cy, 0, 479))
                f3 = float(np.clip(dist_tf[cy_int, cx_int] / (px_per_m * 0.8), 0.10, 0.95))

                _, std_val = cv2.meanStdDev(gray, mask=mask)
                f4 = 1.0 - min(1.0, float(std_val[0][0]) / 40.0)

                dx_err = cx - u0
                dy_err = cy - v0
                f5 = min(1.0, np.sqrt(dx_err**2 + dy_err**2) / np.sqrt(u0**2 + v0**2))

                perim = cv2.arcLength(c, True)
                f6 = (4.0 * np.pi * area) / (perim**2) if perim > 0 else 0.0

                features = np.array([f1, f2, f3, f4, f5, f6])
                is_safe, z_score = self.evaluator.evaluate(features)

                if is_safe and z_score >= 0.0:
                    score = z_score - 0.20 * f5
                    if score > max_score:
                        max_score = score
                        best_target = {"error_x": dx_err, "error_y": dy_err}

            # 3. State Machine & Dynamics
            ex_m, ey_m = 0.0, 0.0
            vz_cmd = 0.0

            if best_target is not None:
                ex_m = best_target["error_x"] / px_per_m
                ey_m = best_target["error_y"] / px_per_m
                h_err = np.sqrt(ex_m**2 + ey_m**2)

                if state in ["SEARCH", "ABORT"]:
                    state = "ALIGN"

                if state == "ALIGN":
                    vz_cmd = 0.0
                    if h_err < 0.15:
                        aligned_counter += 1
                        if aligned_counter >= 3:
                            state = "DESCEND"
                    else:
                        aligned_counter = 0

                elif state == "DESCEND":
                    vz_cmd = -0.32
                    if h_err > 0.40:
                        state = "ALIGN"
                        vz_cmd = 0.0
                        aligned_counter = 0
                    elif drone_pos[2] <= 0.18:
                        state = "TOUCHDOWN"
                        landed = True
                        break
            else:
                state = "ABORT"
                vz_cmd = 0.15  # Climb away from unsafe surface
                aligned_counter = 0
                aborted = True

            # External Disturbance Model (e.g. wind gust)
            drift_x = 0.15 if wind_disturbance else 0.0
            drift_y = -0.10 if wind_disturbance else 0.0

            vx_cmd = pid_x.compute(ex_m, dt) + drift_x
            vy_cmd = pid_y.compute(ey_m, dt) + drift_y

            drone_pos[0] += vx_cmd * dt
            drone_pos[1] += vy_cmd * dt
            drone_pos[2] += vz_cmd * dt
            drone_pos[2] = max(0.0, drone_pos[2])

            history_x.append(drone_pos[0])
            history_y.append(drone_pos[1])
            history_z.append(drone_pos[2])
            time_elapsed += dt

        radial_err = np.sqrt(drone_pos[0]**2 + drone_pos[1]**2)
        return {
            "scenario": scenario_name,
            "landed": landed,
            "aborted": aborted,
            "final_radial_error_cm": radial_err * 100.0,
            "flight_time_s": time_elapsed,
            "final_altitude_m": drone_pos[2]
        }


def run_experiment_battery():
    print("=" * 70)
    print("STAGE 8: RUNNING FLIGHT EXPERIMENTATION BENCHMARK SUITE")
    print("=" * 70)

    harness = DroneExperimentHarness()
    results = []

    # E1: Nominal Clear Pad
    print("[RUNNING E1] Nominal Landing (Clear Environment)...")
    r1 = harness.run_trial("E1_Nominal", wind_disturbance=False, dynamic_obstacle=False)
    results.append(r1)

    # E2: Static Cluttered Region
    print("[RUNNING E2] Static Obstacle / Hazard Rejection...")
    r2 = harness.run_trial("E2_Cluttered_Hazard", wind_disturbance=False, dynamic_obstacle=False)
    results.append(r2)

    # E3: High-Wind Persistent Disturbance
    print("[RUNNING E3] Active Wind Disturbance (Crosswind Drift)...")
    r3 = harness.run_trial("E3_Wind_Disturbance", wind_disturbance=True, dynamic_obstacle=False)
    results.append(r3)

    # E4: Dynamic Mid-Flight Obstacle Intrusion (Safety Abort)
    print("[RUNNING E4] Dynamic Obstacle Intrusion (In-Flight Abort)...")
    r4 = harness.run_trial("E4_Dynamic_Abort", wind_disturbance=False, dynamic_obstacle=True)
    results.append(r4)

    # Display results summary
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("                      EXPERIMENTAL FLIGHT RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))

    out_csv = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[METRICS SAVED] Metrics table exported to: {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    run_experiment_battery()