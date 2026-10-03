import os
import sys
import time
import queue
import threading
import cv2
import numpy as np

# Ensure access to perception and control modules
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT_DIR, "vision"))
sys.path.append(os.path.join(ROOT_DIR, "perceptron"))

from preprocessor import ImagePreprocessor
from landing_detector import LandingZoneDetector
from landing_evaluator import LandingZoneEvaluator
from pid_controller import PIDController


# =====================================================================
# 1. Thread-Safe ROS 2 Topic Middleware Emulator
# =====================================================================
class ROSTopicBus:
    """Simulates the ROS 2 DDS Pub/Sub communication bus."""
    def __init__(self):
        self._topics = {}
        self._lock = threading.Lock()

    def create_topic(self, topic_name: str, maxsize: int = 10):
        with self._lock:
            if topic_name not in self._topics:
                self._topics[topic_name] = []

    def publish(self, topic_name: str, msg):
        with self._lock:
            if topic_name in self._topics:
                for q in self._topics[topic_name]:
                    try:
                        q.put_nowait(msg)
                    except queue.Full:
                        # Drop oldest frame (equivalent to KEEP_LAST history QoS with depth 1)
                        try:
                            _ = q.get_nowait()
                            q.put_nowait(msg)
                        except (queue.Empty, queue.Full):
                            pass

    def subscribe(self, topic_name: str, maxsize: int = 5) -> queue.Queue:
        with self._lock:
            if topic_name not in self._topics:
                self._topics[topic_name] = []
            q = queue.Queue(maxsize=maxsize)
            self._topics[topic_name].append(q)
            return q


bus = ROSTopicBus()
shutdown_event = threading.Event()


# =====================================================================
# 2. Simulated Airframe & World State
# =====================================================================
class DroneWorldState:
    def __init__(self):
        self.lock = threading.Lock()
        # Initial pose: offset by 1.2m West, 0.8m North, at 4.5m altitude
        self.pos = np.array([-1.2, 0.8, 4.5], dtype=np.float64)
        self.pad_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.state = "SEARCH"
        self.landed = False

    def update_kinematics(self, vx: float, vy: float, vz: float, dt: float):
        with self.lock:
            if not self.landed:
                self.pos[0] += vx * dt
                self.pos[1] += vy * dt
                self.pos[2] += vz * dt
                self.pos[2] = max(0.0, self.pos[2])
                if self.pos[2] <= 0.18:
                    self.landed = True
                    self.state = "TOUCHDOWN"

    def get_pose(self):
        with self.lock:
            return self.pos.copy(), self.state, self.landed


world = DroneWorldState()


# =====================================================================
# 3. ROS 2 Nodes (Concurrent Asynchronous Worker Threads)
# =====================================================================

def node_camera_publisher():
    """Publishes raw camera frames on '/camera/image_raw' at 20 Hz."""
    print("[NODE] camera_node started (publishing /camera/image_raw @ 20Hz)")
    u0, v0 = 320.0, 240.0

    while not shutdown_event.is_set():
        pos, _, landed = world.get_pose()
        if landed:
            break

        alt = max(pos[2], 0.20)
        px_per_m = 360.0 / alt

        # Render synthetic camera frame
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        pad_px_x = int(u0 + (0.0 - pos[0]) * px_per_m)
        pad_px_y = int(v0 + (0.0 - pos[1]) * px_per_m)
        pad_radius = int(0.65 * px_per_m)

        if pad_radius > 4:
            cv2.circle(frame, (pad_px_x, pad_px_y), pad_radius, (190, 190, 190), -1)
            cv2.circle(frame, (pad_px_x, pad_px_y), pad_radius, (230, 230, 230), 2)
            cv2.putText(frame, "H", (pad_px_x - int(pad_radius * 0.2), pad_px_y + int(pad_radius * 0.2)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, pad_radius / 40.0), (250, 250, 250), 2)

        # Distribute over ROS bus
        bus.publish("/camera/image_raw", {"image": frame, "alt": alt, "stamp": time.time()})
        time.sleep(0.05)


def node_vision_detector():
    """Subscribes to '/camera/image_raw' and extracts candidate 6D features."""
    print("[NODE] vision_detector_node started (subscribing /camera/image_raw)")
    sub_camera = bus.subscribe("/camera/image_raw", maxsize=2)
    preprocessor = ImagePreprocessor()
    detector = LandingZoneDetector(min_area=150.0, max_area=500000.0)

    while not shutdown_event.is_set():
        try:
            msg = sub_camera.get(timeout=0.2)
        except queue.Empty:
            continue

        frame = msg["image"]
        alt = msg["alt"]
        px_per_m = 360.0 / alt
        u0, v0 = 320.0, 240.0

        _, gray, edges = preprocessor.process(frame)
        candidates, _ = detector.detect_candidates(gray)

        for c in candidates:
            M = cv2.moments(c)
            if M["m00"] <= 1e-4:
                continue
            area = float(M["m00"])
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            expected_area = np.pi * (0.65 * px_per_m)**2
            f1 = float(np.clip(0.80 * (area / max(expected_area, 1.0)), 0.10, 0.95))

            mask = np.zeros((480, 640), dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            cand_edges = cv2.bitwise_and(edges, edges, mask=mask)
            f2 = float(np.count_nonzero(cand_edges)) / max(area, 1.0)

            inv_edges = cv2.bitwise_not(edges)
            dist_tf = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 5)
            f3 = float(np.clip(dist_tf[int(np.clip(cy, 0, 479)), int(np.clip(cx, 0, 639))] / (px_per_m * 0.8), 0.10, 0.95))

            _, std_val = cv2.meanStdDev(gray, mask=mask)
            f4 = 1.0 - min(1.0, float(std_val[0][0]) / 40.0)

            dx = cx - u0
            dy = cy - v0
            f5 = min(1.0, np.sqrt(dx**2 + dy**2) / np.sqrt(u0**2 + v0**2))

            perim = cv2.arcLength(c, True)
            f6 = (4.0 * np.pi * area) / (perim**2) if perim > 0 else 0.0

            feature_vec = np.array([f1, f2, f3, f4, f5, f6])

            bus.publish("/landing/candidate_features", {
                "features": feature_vec,
                "error_x_px": dx,
                "error_y_px": dy,
                "px_per_m": px_per_m
            })


def node_perceptron_classifier():
    """Loads weights and evaluates candidates on '/landing/candidate_features'."""
    print("[NODE] perceptron_node started (subscribing /landing/candidate_features)")
    sub_features = bus.subscribe("/landing/candidate_features", maxsize=5)
    evaluator = LandingZoneEvaluator()

    while not shutdown_event.is_set():
        try:
            msg = sub_features.get(timeout=0.2)
        except queue.Empty:
            continue

        feats = msg["features"]
        is_safe, z_score = evaluator.evaluate(feats)

        bus.publish("/landing/safety_decision", {
            "is_safe": is_safe,
            "z_score": z_score,
            "error_x_px": msg["error_x_px"],
            "error_y_px": msg["error_y_px"],
            "px_per_m": msg["px_per_m"],
            "offset_norm": feats[4]
        })


def node_landing_planner():
    """Runs the FSM with terminal descent latching to avoid FOV edge cropping chatter."""
    print("[NODE] planner_node started (subscribing /landing/safety_decision)")
    sub_decision = bus.subscribe("/landing/safety_decision", maxsize=5)

    fsm_state = "SEARCH"
    aligned_ticks = 0

    while not shutdown_event.is_set():
        try:
            msg = sub_decision.get(timeout=0.2)
        except queue.Empty:
            continue

        is_safe = msg["is_safe"]
        z_score = msg["z_score"]
        px_per_m = msg["px_per_m"]

        ex_m = msg["error_x_px"] / px_per_m
        ey_m = msg["error_y_px"] / px_per_m
        h_err = np.sqrt(ex_m**2 + ey_m**2)

        pos, _, _ = world.get_pose()
        current_alt = pos[2]

        vz_cmd = 0.0

        # Terminal Descent Latch: below 0.70m with <12cm alignment error
        if fsm_state in ["DESCEND", "LAND_LOCK"] and current_alt <= 0.70 and h_err < 0.12:
            fsm_state = "LAND_LOCK"
            vz_cmd = -0.32
        elif is_safe and z_score >= 0.0:
            if fsm_state in ["SEARCH", "ABORT"]:
                fsm_state = "ALIGN"

            if fsm_state == "ALIGN":
                vz_cmd = 0.0
                if h_err < 0.15:
                    aligned_ticks += 1
                    if aligned_ticks >= 3:
                        fsm_state = "DESCEND"
                else:
                    aligned_ticks = 0

            elif fsm_state == "DESCEND":
                vz_cmd = -0.32
                if h_err > 0.35:
                    fsm_state = "ALIGN"
                    vz_cmd = 0.0
                    aligned_ticks = 0
        else:
            if fsm_state != "LAND_LOCK":
                fsm_state = "ABORT"
                vz_cmd = 0.05
                aligned_ticks = 0
            else:
                vz_cmd = -0.32

        bus.publish("/landing/target_error", {
            "ex_m": ex_m,
            "ey_m": ey_m,
            "vz_cmd": vz_cmd,
            "state": fsm_state,
            "h_err": h_err
        })


def node_landing_controller():
    """3-Axis PID controller computing velocity twist commands on '/drone/cmd_vel'."""
    print("[NODE] controller_node started (subscribing /landing/target_error)")
    sub_error = bus.subscribe("/landing/target_error", maxsize=5)

    pid_x = PIDController(kp=1.15, ki=0.04, kd=0.38, max_output=1.4)
    pid_y = PIDController(kp=1.15, ki=0.04, kd=0.38, max_output=1.4)
    dt = 0.05

    while not shutdown_event.is_set():
        try:
            msg = sub_error.get(timeout=0.2)
        except queue.Empty:
            continue

        state = msg["state"]
        ex = msg["ex_m"]
        ey = msg["ey_m"]
        vz = msg["vz_cmd"]

        if state != "TOUCHDOWN":
            vx = pid_x.compute(ex, dt)
            vy = pid_y.compute(ey, dt)
        else:
            vx, vy, vz = 0.0, 0.0, 0.0

        # Update simulated physics
        world.update_kinematics(vx, vy, vz, dt)

        bus.publish("/drone/cmd_vel", {
            "vx": vx, "vy": vy, "vz": vz,
            "state": state, "h_err": msg["h_err"]
        })
        time.sleep(dt)


# =====================================================================
# 4. Main Multi-Threaded Process Orchestrator
# =====================================================================
def main():
    print("=" * 70)
    print("STAGE 7B: ASYNCHRONOUS ROS 2 MULTI-NODE DISTRIBUTED SYSTEM")
    print("=" * 70)

    # Launch all 5 nodes concurrently on distinct OS threads
    threads = [
        threading.Thread(target=node_camera_publisher, daemon=True),
        threading.Thread(target=node_vision_detector, daemon=True),
        threading.Thread(target=node_perceptron_classifier, daemon=True),
        threading.Thread(target=node_landing_planner, daemon=True),
        threading.Thread(target=node_landing_controller, daemon=True)
    ]

    for t in threads:
        t.start()

    print("\n[*] All ROS 2 Nodes Active and Communicating Over Pub/Sub Bus.\n")

    sub_feedback = bus.subscribe("/drone/cmd_vel", maxsize=5)
    start_t = time.time()

    try:
        while True:
            try:
                msg = sub_feedback.get(timeout=0.5)
            except queue.Empty:
                continue

            pos, state, landed = world.get_pose()
            elapsed = time.time() - start_t

            if landed:
                print("\n" + "=" * 70)
                print(f"[ROS 2 TOUCHDOWN] Landed successfully at T = {elapsed:.2f}s!")
                final_err = np.sqrt(pos[0]**2 + pos[1]**2)
                print(f"Final Radial Accuracy: {final_err * 100:.2f} cm from target center.")
                print("=" * 70)
                break

            # Telemetry stream to terminal
            print(f"[ROS 2 TELEMETRY] T={elapsed:4.1f}s | FSM={msg['state']:<9} | "
                  f"Alt={pos[2]:.2f}m | Pos=({pos[0]:+.2f}, {pos[1]:+.2f})m | "
                  f"CmdVel=({msg['vx']:+.2f}, {msg['vy']:+.2f}, {msg['vz']:+.2f}) m/s")

            time.sleep(0.15)

    except KeyboardInterrupt:
        print("\n[TERMINATING] Shutting down ROS 2 nodes...")
    finally:
        shutdown_event.set()
        time.sleep(0.5)
        print("[SUCCESS] All ROS 2 worker nodes safely terminated.")


if __name__ == "__main__":
    main()