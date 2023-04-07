#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import keyboard
import cv2
from cv_bridge import CvBridge
from time import sleep
import numpy as np
import sys
import termios
from plate_detector import PlateDetector


class ImitationTrainer:
    def __init__(self):
        self.pub = rospy.Publisher("/R1/cmd_vel", Twist)
        self.sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)

        self.images = []
        self.velocities = []
        self.file_path = "/home/fizzer/ros_ws/src/controller_pkg/cnn_trainer/data/"
        self.recording = False

        # self.rate = rospy.Rate(25)
        self.move = Twist()
        self.move.linear.x = 0.0
        self.move.angular.z = 0.0

        # DO NOT CHANGE THESE!
        self.FORWARD_SPEED = 0.25
        self.TURN_SPEED = 1.0

        # sleep time for one-off button presses
        self.SLEEP_TIME = 0.25

        # should be a factor of 720 and 1280
        self.COMPRESSION_KERN = 8
        self.image_counter = 0
        self.checkpoint_idx = 0

        self.bridge = CvBridge()
        self.print_flag = True

        # pedestrian recognition test
        self.pedestrian_data_flag = True
        self.r, self.c = 400, 540
        self.width = 200
        self.height = 225

        self.prev_mean = 0.0
        self.current_mean = 0.0

        # plate detection
        self.plate_detector = PlateDetector()

        # testing and other
        self.TEST_CAPTURE = False

    def callback(self, data):
        recording_status = "PAUSED."
        if self.recording:
            recording_status = "RECORDING!"
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

            # testing only

            # img, approx_contours = self.plate_detector._detect_area(frame)
            # approx_contours = np.array(approx_contours, dtype=int)
            # print(approx_contours)
            # if approx_contours.shape[0] != 0:
            #     img_copy = np.copy(img)
            #     approx_img = cv2.drawContours(
            #         img_copy, [approx_contours], -1, (0, 255, 0), 3
            #     )
            #     print("contours detected")
            #     cv2.imshow("plate", approx_img)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

            # if plate.shape[0] != 0:
            #     print("Plate detected!")
            #     cv2.imshow("plate", plate)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            # else:
            #     pass

            pedestrian_frame = frame[
                self.r : self.r + self.height + 1, self.c : self.c + self.width + 1
            ]
            mean = round(np.mean(pedestrian_frame), 2)
            # print("mean: ", mean)
            if self.prev_mean > (mean + 0.1):
                # print("PEDESTRIAN CROSSING!")
                pass

            self.prev_mean = mean

            if self.TEST_CAPTURE:
                # no compression or grayscale or anything
                pass
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = frame[..., np.newaxis]
                frame = self.compress(frame, self.COMPRESSION_KERN)

            self.images.append(frame)
            self.velocities.append((self.move.linear.x, self.move.angular.z))

            self.image_counter += 1

        if self.print_flag:
            print(
                "{} Saved {} images".format(recording_status, self.image_counter),
                end="\r",
            )

    def check_record_save(self):
        if keyboard.is_pressed("s"):
            self.print_flag = False
            rospy.sleep(self.SLEEP_TIME)
            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
            yn = input("Do you want to save images and vels to data/? [y/n]: ")
            print(yn)

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

    def check_record_rollback(self):
        if keyboard.is_pressed("z"):
            self.print_flag = False
            rospy.sleep(self.SLEEP_TIME)
            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
            yn = input("Do you want to roll back to previous checkpoint? [y/n]: ")
            print(yn)

            if yn == "y":
                self.images = self.images[: self.checkpoint_idx]
                self.velocities = self.velocities[: self.checkpoint_idx]
                self.image_counter = self.checkpoint_idx
                print("Rolled back to previous checkpoint.")

            elif yn == "n":
                print("Did not roll back.")
            self.print_flag = True

    def check_checkpoint(self):
        if keyboard.is_pressed("x"):
            rospy.sleep(self.SLEEP_TIME)
            self.checkpoint_idx = self.image_counter
            print("Set checkpoint at {} images.".format(self.checkpoint_idx))

    def check_record_start(self):
        if keyboard.is_pressed("q"):
            rospy.sleep(self.SLEEP_TIME)
            self.recording = True

    def check_record_pause(self):
        if keyboard.is_pressed("e"):
            rospy.sleep(self.SLEEP_TIME)
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
            self.check_record_rollback()
            self.check_checkpoint()

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
                self.move.linear.x = 0.0
                self.move.angular.z = 0.0

            self.pub.publish(self.move)


if __name__ == "__main__":
    rospy.init_node("IMITATION_TRAINER")
    it = ImitationTrainer()
    it.start()
