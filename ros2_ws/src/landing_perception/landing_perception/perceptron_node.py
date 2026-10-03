import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import os


class PerceptronClassifierNode(Node):
    def __init__(self):
        super().__init__('perceptron_classifier_node')

        # Locate trained model parameters
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(pkg_dir, "../../../../perceptron/trained_model.npz"))

        if not os.path.exists(model_path):
            self.get_logger().error(f"Cannot find model weights at: {model_path}")
            raise FileNotFoundError(model_path)

        data = np.load(model_path, allow_pickle=True)
        self.weights = data['weights']
        self.bias = float(data['bias'])
        self.scaler_mean = data['scaler_mean']
        self.scaler_scale = data['scaler_scale']

        self.get_logger().info(
            f"Perceptron Loaded. Bias: {self.bias:.4f}, Weights: {np.round(self.weights, 4)}"
        )

        # Subscriber: Receives raw features + centroid offsets [f1..f6, ex, ey]
        self.sub_features = self.create_subscription(
            Float32MultiArray,
            '/landing/candidate_features',
            self.features_callback,
            10
        )

        # Publisher: Publishes classification results [is_safe (1/0), z_score, ex, ey]
        self.pub_decision = self.create_publisher(
            Float32MultiArray,
            '/landing/safety_decision',
            10
        )

    def features_callback(self, msg: Float32MultiArray):
        raw_data = np.array(msg.data, dtype=np.float64)
        if len(raw_data) < 8:
            return

        features = raw_data[:6]
        ex = raw_data[6]
        ey = raw_data[7]

        # Z-Score Standardization
        scaled_features = (features - self.scaler_mean) / self.scaler_scale

        # Perceptron linear forward pass: z = w^T * x' + b
        z = float(np.dot(scaled_features, self.weights) + self.bias)
        is_safe = 1.0 if z >= 0.0 else 0.0

        # Publish decision message
        out_msg = Float32MultiArray()
        out_msg.data = [is_safe, z, ex, ey]
        self.pub_decision.publish(out_msg)

        status_str = "SAFE" if is_safe == 1.0 else "UNSAFE"
        self.get_logger().debug(f"Candidate Evaluation: {status_str} (z={z:+.4f}) | ex={ex:.1f}, ey={ey:.1f}")


def main(args=None):
    rclpy.init(args=args)
    node = PerceptronClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()