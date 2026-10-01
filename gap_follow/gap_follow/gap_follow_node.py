#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class GapFollowNode(Node):
    """
    Implement Gap Following on the car
    """

    def __init__(self):
        super().__init__('gap_follow_node')

        # Topics
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Subscribe to LIDAR
        self.scan_sub = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.lidar_callback,
            10
        )

        # Publish drive commands
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )

        self.angle = 0.0
        self.speed = 0.0
        self.ranges = []

        self.min_indx = 0
        self.max_indx = 0
        self.proc_ranges = []


    def preprocess_lidar(self, ranges, angle_min, angle_increment):
        """
        Preprocess the LiDAR scan.

        Only use points from -70 to +70 degrees.
        Invalid values are removed and very large values are limited.
        """

        proc_ranges = np.array(ranges, dtype=float)

        # Replace invalid readings
        proc_ranges = np.nan_to_num(
            proc_ranges,
            nan=0.0,
            posinf=3.0,
            neginf=0.0
        )

        # Limit maximum distance to 3 meters
        proc_ranges = np.clip(proc_ranges, 0.0, 3.0)

        # Small smoothing window
        window_size = 3
        kernel = np.ones(window_size) / window_size
        proc_ranges = np.convolve(proc_ranges, kernel, mode='same')

        # Only look at the front 140 degrees
        min_angle = np.radians(-70)
        max_angle = np.radians(70)

        min_indx = int((min_angle - angle_min) / angle_increment)
        max_indx = int((max_angle - angle_min) / angle_increment)

        # Make sure indexes stay inside the array
        min_indx = max(0, min_indx)
        max_indx = min(len(proc_ranges) - 1, max_indx)

        self.min_indx = min_indx
        self.max_indx = max_indx

        return proc_ranges, min_indx, max_indx


    def find_max_gap(self, free_space_ranges):
        """
        Find the longest continuous section of non-zero LiDAR values.
        """

        min_indx = self.min_indx
        max_indx = self.max_indx

        start = min_indx
        end = min_indx

        current_start = None
        longest_duration = 0

        for i in range(min_indx, max_indx + 1):

            if free_space_ranges[i] > 0.0:

                if current_start is None:
                    current_start = i

            else:

                if current_start is not None:

                    current_duration = i - current_start

                    if current_duration > longest_duration:
                        longest_duration = current_duration
                        start = current_start
                        end = i - 1

                    current_start = None

        # Handle a gap that continues to the end
        if current_start is not None:

            current_duration = max_indx - current_start + 1

            if current_duration > longest_duration:
                start = current_start
                end = max_indx

        self.proc_ranges = free_space_ranges

        return start, end


    def find_best_point(
        self,
        start_i,
        end_i,
        ranges,
        angle_min,
        angle_increment
    ):
        """
        Choose the middle of the largest gap.

        The middle of the gap is usually more stable than steering toward
        one single maximum LiDAR reading.
        """

        best_index = (start_i + end_i) // 2

        best_angle = angle_min + best_index * angle_increment

        # Limit steering angle
        max_steering_angle = np.radians(35)

        best_angle = np.clip(
            best_angle,
            -max_steering_angle,
            max_steering_angle
        )

        return best_angle


    def lidar_callback(self, data):
        """
        Process each LiDAR scan using the Follow Gap algorithm
        and publish an AckermannDriveStamped message.
        """

        ranges = data.ranges
        angle_min = data.angle_min
        angle_increment = data.angle_increment

        proc_ranges, min_indx, max_indx = self.preprocess_lidar(
            ranges,
            angle_min,
            angle_increment
        )

        # ---------------------------------------------------------
        # Find closest point
        # ---------------------------------------------------------

        front_ranges = proc_ranges[min_indx:max_indx + 1]

        # Ignore zero readings when searching for closest obstacle
        valid_ranges = np.where(
            front_ranges > 0.05,
            front_ranges,
            np.inf
        )

        closest_local_index = np.argmin(valid_ranges)
        closest_index = min_indx + closest_local_index
        closest_distance = proc_ranges[closest_index]

        # If no valid point exists, just return
        if not np.isfinite(valid_ranges[closest_local_index]):
            return

        # ---------------------------------------------------------
        # Dynamic safety bubble
        # ---------------------------------------------------------

        # Bubble becomes larger as speed increases
        bubble_radius_m = 0.30 + (0.15 * self.speed)

        # Convert physical bubble size into an angle
        bubble_angle = np.arctan2(
            bubble_radius_m,
            max(closest_distance, 0.1)
        )

        bubble_size = int(
            bubble_angle / angle_increment
        )

        # ---------------------------------------------------------
        # Remove points inside safety bubble
        # ---------------------------------------------------------

        bubble_start = max(
            min_indx,
            closest_index - bubble_size
        )

        bubble_end = min(
            max_indx,
            closest_index + bubble_size
        )

        proc_ranges[
            bubble_start:bubble_end + 1
        ] = 0.0

        # ---------------------------------------------------------
        # Find maximum free-space gap
        # ---------------------------------------------------------

        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # ---------------------------------------------------------
        # Find best point inside the gap
        # ---------------------------------------------------------

        target_angle = self.find_best_point(
            gap_start,
            gap_end,
            proc_ranges,
            angle_min,
            angle_increment
        )

        # ---------------------------------------------------------
        # Set speed based on steering angle
        # ---------------------------------------------------------

        abs_angle = abs(target_angle)

        if abs_angle < np.radians(10):
            speed = 1.5

        elif abs_angle < np.radians(20):
            speed = 1.0

        else:
            speed = 0.5

        # ---------------------------------------------------------
        # Smooth steering
        # ---------------------------------------------------------

        smoothing = 0.35

        smoothed_angle = (
            (1.0 - smoothing) * self.angle
            + smoothing * target_angle
        )

        self.angle = smoothed_angle
        self.speed = speed

        # ---------------------------------------------------------
        # Publish drive message
        # ---------------------------------------------------------

        drive_msg = AckermannDriveStamped()

        drive_msg.drive.steering_angle = float(self.angle)
        drive_msg.drive.speed = float(self.speed)

        self.drive_pub.publish(drive_msg)


def main(args=None):

    rclpy.init(args=args)

    print("GapFollow Initialized")

    gap_follow_node = GapFollowNode()

    rclpy.spin(gap_follow_node)

    gap_follow_node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()