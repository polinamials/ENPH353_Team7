#!/usr/bin/env python3
import numpy as np
import rospy
import tensorflow as tf
from tensorflow import keras
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

# TODO: so far have 18000 data, but the driving wasn't always perfect.
# Gather probably 36000 data with better driving


class CNNDriver:
    def __init__(self):
        self.pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
        self.sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.callback, queue_size=10
        )
        self.driver_model = tf.keras.models.load_model(
            "/home/fizzer/ros_ws/cnn_trainer/trained_model"
        )
        self.move = Twist()
        self.move.linear.x = 0.0
        self.move.angular.z = 0.0
        self.actions_to_vels = np.array([(0.25, 0.0), (0.25, 1.0), (0.25, -1.0)])
        self.COMPRESSION_KERN = 8
        self.bridge = CvBridge()

        # depends on how long it takes to predict a model
        self.wait_in_ms = 30
        self.hz = 1000 // self.wait_in_ms
        #self.rate = rospy.Rate(self.hz)
        # This is in milliseconds
        self.start_time = rospy.Time().now().to_nsec() // 1000000

    def callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        # print(frame.shape)
        compressed_frame = self.compress(frame, self.COMPRESSION_KERN)
        gray_frame = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = gray_frame[np.newaxis, ...]
        gray_frame = gray_frame[..., np.newaxis]
        gray_frame = gray_frame / 255
        # print(compressed_frame.shape)
        #self.predict_vel(gray_frame)
        self.move.linear.x = 0.0
        self.move.angular.z = 0.0

    def compress(self, img, kern):
        h, w, d = img.shape
        return np.array(
            img.reshape(h // kern, kern, w // kern, kern, d).mean(axis=(1, 3)),
            dtype="uint8",
        )

    def predict_vel(self, frame):
        """self.move.linear.x = 0.0
        self.move.angular.z = 0.0
        self.pub.publish(self.move)"""
        arr = self.driver_model(frame)
        x, z = self.actions_to_vels[np.argmax(arr)]
        # print("predicting: ", x, z)
        self.move.linear.x = x
        self.move.angular.z = z
        self.pub.publish(self.move)


if __name__ == "__main__":
    rospy.init_node("CNN_DRIVER")
    it = CNNDriver()
    # it.rate.sleep()
    rospy.spin()
