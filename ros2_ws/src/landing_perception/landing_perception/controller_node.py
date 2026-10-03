import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
import numpy as np


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, max_out: float = 1.2, max_i: float = 0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.max_i = max_i
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        p = self.kp * error
        self.integral = float(np.clip(self.integral + error * dt, -self.max_i, self.max_i))
        i = self.ki * self.integral
        d = 0.0 if self.first_run else self.kd * (error - self.prev_error) / dt
        self.first_run = False
        self.prev_error = error
        return float(np.clip(p + i + d, -self.max_out, self.max_out))


class LandingControllerNode(Node):
    def __init__(self):
        super().__init__('landing_controller_node')

        self.pid_x = PIDController(kp=1.15, ki=0.04, kd=0.38, max_out=1.4)
        self.pid_y = PIDController(kp=1.15, ki=0.04, kd=0.38, max_out=1.4)
        self.last_time = self.get_clock().now()

        # Subscriber: Target Errors
        self.sub_error = self.create_subscription(
            PointStamped,
            '/landing/target_error',
            self.error_callback,
            10
        )

        # Publisher: Twist velocity setpoints
        self.pub_cmd_vel = self.create_publisher(
            Twist,
            '/drone/cmd_vel',
            10
        )

        self.get_logger().info("Landing PID Controller Initialized.")

    def error_callback(self, msg: PointStamped):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if dt <= 0.0 or dt > 0.5:
            dt = 0.1

        ex = msg.point.x
        ey = msg.point.y
        vz_cmd = msg.point.z

        # Compute PID outputs
        vx_cmd = self.pid_x.compute(ex, dt)
        vy_cmd = self.pid_y.compute(ey, dt)

        twist = Twist()
        twist.linear.x = vx_cmd
        twist.linear.y = vy_cmd
        twist.linear.z = vz_cmd
        self.pub_cmd_vel.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = LandingControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()