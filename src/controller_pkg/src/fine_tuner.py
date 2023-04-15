#!/usr/bin/env python3
import numpy as np
import rospy
import tensorflow as tf
from tensorflow import keras
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import keyboard
import termios
import sys


class FineTuner:
    def __init__(self):
        self.pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
        self.sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.callback, queue_size=10
        )
        self.driver_model = tf.keras.models.load_model(
            "/home/fizzer/ros_ws/src/controller_pkg/cnn_trainer/inner_loop_rm_straight_bias_2"
        )
        self.move = Twist()
        self.move.linear.x = 0.0
        self.move.angular.z = 0.0
        self.actions_to_vels = np.array([(0.25, 0.0), (0.25, 1.0), (0.25, -1.0)])
        self.COMPRESSION_KERN = 8
        self.bridge = CvBridge()
        self.model_pub = False
        self.record = False
        self.TURN_SPEED = 1.0

        self.images = []
        self.velocities = []
        self.file_path = "/home/fizzer/ros_ws/src/controller_pkg/cnn_trainer/data"

        self.SLEEP_TIME = 0.25
        self.image_counter = 0
        self.print_flag = True

    def callback(self, data):
        self.check_record_save()
        if keyboard.is_pressed("a"):
            self.model_pub = False
            self.record = True
            self.move.angular.z = self.TURN_SPEED
            self.pub.publish(self.move)
            self.model_pub = True
        elif keyboard.is_pressed("d"):
            self.model_pub = False
            self.record = True
            self.move.angular.z = -self.TURN_SPEED
            self.pub.publish(self.move)
            self.model_pub = True
        else:
            self.model_pub = True
            self.record = False
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        compressed_frame = self.compress(frame, self.COMPRESSION_KERN)
        gray_frame = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = gray_frame[..., np.newaxis]

        if self.record:
            recording_status = "RECORDING!"
            self.images.append(gray_frame)
            self.velocities.append((self.move.linear.x, self.move.angular.z))
            self.image_counter += 1
        else:
            recording_status = "PAUSED."

        if self.print_flag:
            print(
                "{} Saved {} images".format(recording_status, self.image_counter),
                end="\r",
            )

        gray_frame = gray_frame[np.newaxis, ...]
        gray_frame = gray_frame / 255

        if self.model_pub:
            self.predict_vel(gray_frame)
        else:
            pass

    def compress(self, img, kern):
        h, w, d = img.shape
        return np.array(
            img.reshape(h // kern, kern, w // kern, kern, d).mean(axis=(1, 3)),
            dtype="uint8",
        )

    def predict_vel(self, frame):
        arr = self.driver_model(frame)
        x, z = self.actions_to_vels[np.argmax(arr)]
        self.move.linear.x = x
        self.move.angular.z = z
        # rospy.sleep(0.25)
        self.pub.publish(self.move)

    def check_record_save(self):
        if keyboard.is_pressed("s"):
            # self.sub.unregister()
            self.move.linear.x = 0.0
            self.move.angular.z = 0.0
            self.pub.publish(self.move)
            self.print_flag = False
            rospy.sleep(self.SLEEP_TIME)
            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
            yn = input("Do you want to save images and vels to data/? [y/n]: ")

            if yn == "y":
                i = input("Enter the name of the images file (excluding extension): ")
                v = input("Enter the name of the vels file (excluding extension): ")

                images_name = "{}.npy".format(i)
                vels_name = "{}.npy".format(v)

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

                self.images = []
                self.velocities = []
                self.image_counter = 0
            elif yn == "n":
                print("File was not saved.")
            self.print_flag = True


if __name__ == "__main__":
    # rospy.sleep(10)
    rospy.init_node("FINE_TUNER")
    ft = FineTuner()
    rospy.spin()
    # ft.start()
