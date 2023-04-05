#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import keyboard
import cv2
from cv_bridge import CvBridge
from time import sleep
import numpy as np


"""
For training we need:

10 full laps, including the starting exit
10 times around each corner


"""


class ImitationTrainer:
    def __init__(self):
        self.pub = rospy.Publisher("/R1/cmd_vel", Twist)
        self.sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)

        self.images = []
        self.velocities = []
        self.file_count = 0
        self.file_path = "/home/fizzer/ros_ws/cnn_trainer/data/"
        self.recording = False

        # self.rate = rospy.Rate(25)
        self.move = Twist()
        self.move.linear.x = 0.0
        self.move.angular.z = 0.0

        # you can change these constants
        self.FORWARD_SPEED = 0.25
        self.TURN_SPEED = 1.0
        self.SLEEP_TIME = 0.250

        # should be a factor of 720 and 1280
        self.COMPRESSION_KERN = 8
        self.image_counter = 0
        # self.cam_freq = (1.0 / self.IMAGES_PER_SECOND) * 10.0**9
        # print(self.cam_freq)

        self.bridge = CvBridge()

    def callback(self, data):
        recording_status = "PAUSED."
        if self.recording:
            recording_status = "RECORDING!"
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[..., np.newaxis]

            self.images.append(self.compress(frame, self.COMPRESSION_KERN))
            self.velocities.append((self.move.linear.x, self.move.angular.z))
            self.image_counter += 1

        print(
            "{} Saved {} images".format(recording_status, self.image_counter), end="\r"
        )

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

    def check_record_start(self):
        if keyboard.is_pressed("q"):
            rospy.sleep(self.SLEEP_TIME)
            # print("Starting the recording.", end="\r")
            self.recording = True

    def check_record_pause(self):
        if keyboard.is_pressed("e"):
            rospy.sleep(self.SLEEP_TIME)
            # print("Pausing the recording.", end="\r")
            self.recording = False

    def compress(self, img, kern):
        h, w, d = img.shape
        return np.array(
            img.reshape(h // kern, kern, w // kern, kern, d).mean(axis=(1, 3)),
            dtype="uint8",
        )

    def start(self):
        while not rospy.is_shutdown():
            self.check_record_save()
            self.check_record_start()
            self.check_record_pause()

            if keyboard.is_pressed("w"):
                if keyboard.is_pressed("a"):
                    self.move.angular.z = self.TURN_SPEED

                elif keyboard.is_pressed("d"):
                    self.move.angular.z = -self.TURN_SPEED

                else:
                    self.move.linear.x = self.FORWARD_SPEED
                    self.move.angular.z = 0.0

            elif keyboard.is_pressed("a"):
                if keyboard.is_pressed("w"):
                    self.move.linear.x = self.FORWARD_SPEED

                else:
                    self.move.angular.z = self.TURN_SPEED
                    self.move.linear.x = 0.0

            elif keyboard.is_pressed("d"):
                if keyboard.is_pressed("w"):
                    self.move.linear.x = self.FORWARD_SPEED

                else:
                    self.move.angular.z = -self.TURN_SPEED
                    self.move.linear.x = 0.0

            else:
                """if keyboard.is_pressed("j"):
                    rospy.sleep(self.SLEEP_TIME)
                    self.FORWARD_SPEED += 0.01
                    print("New linear speed is: ", self.FORWARD_SPEED)
                elif keyboard.is_pressed("h"):
                    rospy.sleep(self.SLEEP_TIME)
                    self.FORWARD_SPEED -= 0.01
                    print("New linear speed is: ", self.FORWARD_SPEED)
                elif keyboard.is_pressed("l"):
                    rospy.sleep(self.SLEEP_TIME)
                    self.TURN_SPEED += 0.1
                    print("New angular speed is: ", self.TURN_SPEED)
                elif keyboard.is_pressed("k"):
                    rospy.sleep(self.SLEEP_TIME)
                    self.TURN_SPEED -= 0.1
                    print("New angular speed is: ", self.TURN_SPEED)
                else:"""
                self.move.linear.x = 0.0
                self.move.angular.z = 0.0

            self.pub.publish(self.move)


if __name__ == "__main__":
    rospy.init_node("IMITATION_TRAINER")
    it = ImitationTrainer()
    it.start()
