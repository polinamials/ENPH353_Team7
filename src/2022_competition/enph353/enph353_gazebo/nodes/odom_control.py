#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, Point, Quaternion, Pose
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import keyboard
import cv2
from cv_bridge import CvBridge
from time import sleep
import numpy as np


class OdomControl:
    def __init__(self):
        self.pub = rospy.Publisher("/R1/cmd_vel", Twist)
        self.sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
        self.sub_odom = rospy.Subscriber("/R1/odom", Odometry, self.odom_cb)

        self.images = []
        self.velocities = []
        self.file_count = 0
        self.file_path = "/home/fizzer/ros_ws/cnn_trainer/data/"
        self.recording = False

        # self.rate = rospy.Rate(25)
        self.move = Twist()
        self.move.linear.x = 0.0
        self.move.angular.z = 0.0

        # self.rate = rospy.Rate(0.5)

        # you can change these constants
        self.FORWARD_SPEED = 0.25
        self.TURN_SPEED = 1.0
        self.SLEEP_TIME = 0.250

        # should be a factor of 720 and 1280
        self.COMPRESSION_KERN = 8
        self.image_counter = 0

        self.bridge = CvBridge()
        self.x = 0.0
        self.y = -10.0
        self.angle = 0.0

    def callback(self, data):
        recording_status = "PAUSED."
        if self.recording:
            recording_status = "RECORDING!"
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

            self.images.append(self.compress(frame, self.COMPRESSION_KERN))
            self.velocities.append((self.move.linear.x, self.move.angular.z))
            self.image_counter += 1

        """print(
            "{} Saved {} images".format(recording_status, self.image_counter), end="\r"
        )"""

    def odom_cb(self, msg):
        """print("position")
        print("x ", msg.pose.pose.position.x)
        print("y", msg.pose.pose.position.y)
        print("orientation")
        print("z ", msg.pose.pose.orientation.z)"""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.angle = msg.pose.pose.orientation.z

    def check_record_save(self):
        if keyboard.is_pressed("c"):
            rospy.sleep(self.SLEEP_TIME)
            images_name = "imgs_{}.npy".format(self.file_count)
            vels_name = "vels_{}.npy".format(self.file_count)

            print(
                "Saving files {} and {} to {}".format(
                    images_name, vels_name, self.file_path
                )
            )

            np.save(
                "{}{}".format(self.file_path, images_name),
                self.images,
            )
            np.save(
                "{}{}".format(self.file_path, vels_name),
                self.velocities,
            )

            self.file_count += 1
            # THESE ARRAYS NEED TO BE RESET
            # so that each next save is not cumulative
            self.images = []
            self.velocities = []
            self.image_counter = 0

    def compress(self, img, kern):
        h, w, d = img.shape
        return np.array(
            img.reshape(h // kern, kern, w // kern, kern, d).mean(axis=(1, 3)),
            dtype="uint8",
        )

    def go(self):
        """while round(self.x, 2) != -0.91:
            self.move.linear.x = 0.25
            self.move.angular.z = 0.0

            self.pub.publish(self.move)

        while round(self.angle, 2) != 0.72:
            self.move.linear.x = 0.25
            self.move.angular.z = 1.0

            self.pub.publish(self.move)

        while round(self.y, 2) != -0.94:
            self.move.linear.x = 0.25
            self.move.angular.z = 0.0

            self.pub.publish(self.move)

        while round(self.angle, 2) != 0.00:
            self.move.linear.x = 0.25
            self.move.angular.z = 1.0

            self.pub.publish(self.move)

        while round(self.x, 2) != 0.94:
            self.move.linear.x = 0.25
            self.move.angular.z = 0.0

            self.pub.publish(self.move)

        while round(self.angle, 2) != -0.69:
            self.move.linear.x = 0.25
            self.move.angular.z = 1.0

            self.pub.publish(self.move)

        while round(self.y, 2) != 0.94:
            self.move.linear.x = 0.25
            self.move.angular.z = 0.0

            self.pub.publish(self.move)

        while round(self.angle, 2) != -1.01:
            self.move.linear.x = 0.25
            self.move.angular.z = 1.0

            self.pub.publish(self.move)

        while round(self.x, 2) != -0.91:
            self.move.linear.x = 0.25
            self.move.angular.z = 0.0

            self.pub.publish(self.move)

        self.move.linear.x = 0.0
        self.move.angular.z = 0.0
        self.pub.publish(self.move)

        rospy.sleep(0.5)"""

        """for i in range(5):
            self.move.linear.x = 0.25
            self.move.angular.z = 0.0
            self.pub.publish(self.move)
            rospy.sleep(0.25)

        for i in range(5):
            self.move.linear.x = 0.0
            self.move.angular.z = np.pi / 2
            self.pub.publish(self.move)
            rospy.sleep(0.25)

        self.move.linear.x = 0.0
        self.move.angular.z = 0.0
        self.pub.publish(self.move)"""
        x = 0


if __name__ == "__main__":
    rospy.init_node("ODOM_CONTROL")
    it = OdomControl()
    # it.start()
    it.go()
