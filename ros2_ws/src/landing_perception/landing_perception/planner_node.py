import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
import numpy as np


class LandingPlannerNode(Node):
    def __init__(self):
        super().__init__('landing_planner_node')

        self.current_state = "SEARCH"
        self.drone_altitude = 4.5  # Nominal altitude default in meters
        self.aligned_ticks = 0

        # Subscriber: Safety decision stream
        self.sub_decision = self.create_subscription(
            Float32MultiArray,
            '/landing/safety_decision',
            self.decision_callback,
            10
        )

        # Publisher: Metric target position errors (Ex, Ey, Ez)
        self.pub_target_error = self.create_publisher(
            PointStamped,
            '/landing/target_error',
            10
        )

        self.get_logger().info("Landing Planner & State Machine Node Initialized.")

    def decision_callback(self, msg: Float32MultiArray):
        is_safe = bool(msg.data[0])
        z_score = float(msg.data[1])
        ex_px = float(msg.data[2])
        ey_px = float(msg.data[3])

        # Pinhole conversion: pixel errors to meters based on current altitude
        px_per_m = 360.0 / max(self.drone_altitude, 0.20)
        ex_m = ex_px / px_per_m
        ey_m = ey_px / px_per_m
        h_error = np.sqrt(ex_m**2 + ey_m**2)

        vz_target = 0.0

        if is_safe and z_score >= 0.001:
            if self.current_state in ["SEARCH", "ABORT"]:
                self.current_state = "ALIGN"
                self.get_logger().info("Target detected & SAFE -> State: ALIGN")

            if self.current_state == "ALIGN":
                vz_target = 0.0
                if h_error < 0.15:
                    self.aligned_ticks += 1
                    if self.aligned_ticks >= 5:
                        self.current_state = "DESCEND"
                        self.get_logger().info("Drone Aligned -> State: DESCEND")
                else:
                    self.aligned_ticks = 0

            elif self.current_state == "DESCEND":
                vz_target = -0.30  # Descent rate 30 cm/s
                if h_error > 0.40:
                    self.current_state = "ALIGN"
                    self.aligned_ticks = 0
                    self.get_logger().warn("Centering Drift -> Reverting to ALIGN")
        else:
            self.current_state = "ABORT"
            vz_target = 0.05
            self.aligned_ticks = 0

        # Publish target error vector
        pt_msg = PointStamped()
        pt_msg.header.stamp = self.get_clock().now().to_msg()
        pt_msg.header.frame_id = self.current_state
        pt_msg.point.x = ex_m
        pt_msg.point.y = ey_m
        pt_msg.point.z = vz_target
        self.pub_target_error.publish(pt_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LandingPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()