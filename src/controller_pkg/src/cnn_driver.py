#!/usr/bin/env python3
import numpy as np
import rospy
import tensorflow as tf
from tensorflow import keras
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
from plate_detector import PlateDetector


class CNNDriver:
    def __init__(self):
        # driving
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
        self.cam_sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.callback, queue_size=10
        )
        self.driver_model = tf.keras.models.load_model(
            "/home/fizzer/ros_ws/src/controller_pkg/cnn_trainer/trained_model_rm_straight_bias"
        )
        self.move = Twist()
        self.move.linear.x = 0.0
        self.move.angular.z = 0.0
        self.actions_to_vels = np.array([(0.25, 0.0), (0.25, 1.0), (0.25, -1.0)])
        self.COMPRESSION_KERN = 8
        self.bridge = CvBridge()

        # crosswalk detection
        self.lower = np.array([0, 0, 250])
        self.upper = np.array([5, 5, 255])
        self.bottom_crop = -50
        self.red = 0.0

        # pedestrian crossing
        self.check_crosswalk_flag = True
        self.RED_THRESH = 3.5
        self.r, self.c = 400, 540
        self.width = 200
        self.height = 225
        self.PED_THRESH = 0.1
        self.prev_mean = 0.0
        self.pedestrian_crossed = False
        self.cool_off_start_time = rospy.Time.now().to_sec()
        self.PED_COOL_OFF = 2

        # score tracker
        self.score_pub = rospy.Publisher("/license_plate", String, queue_size=1)
        # time trials only
        # self.timer = rospy.Timer(rospy.Duration(60), self.stop_timer)
        # self.started_timer = False
        # self.stopped_timer = False

        # plate detection
        self.plate_detector = PlateDetector()
        self.PLATE_COOL_OFF = 0.5

    def callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

        # TODO
        # while its passing the plate, it gets a bunch of images of it.
        # We predict each one and store the PROBABILITEIS
        # Then we have some sort of timer that says:
        # if you haven't seen a plate in half a second, means you passed it.
        # now choose the highest overall prediction FOR EACH SYMBOL
        # print plate number

        ##########

        plate_number = self.plate_detector.read_plate(frame)
        if plate_number:
            print("plate number: ", plate_number)

        ##########

        if rospy.Time.now().to_sec() - self.cool_off_start_time >= self.PED_COOL_OFF:
            self.check_crosswalk(frame)

        if self.red <= self.RED_THRESH:
            compressed_frame = self.compress(frame, self.COMPRESSION_KERN)
            gray_frame = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2GRAY)
            gray_frame = gray_frame[np.newaxis, ...]
            gray_frame = gray_frame[..., np.newaxis]
            gray_frame = gray_frame / 255
            self.predict_vel(gray_frame)
        else:
            # TODO make this a function
            if not self.pedestrian_crossed:
                self.move.linear.x = 0.0
                self.move.angular.z = 0.0
                self.vel_pub.publish(self.move)

                # TODO remove magic number
                rospy.sleep(0.1)

                pedestrian_frame = frame[
                    self.r : self.r + self.height + 1, self.c : self.c + self.width + 1
                ]

                current_mean = round(np.mean(pedestrian_frame), 2)
                if self.prev_mean > (current_mean + self.PED_THRESH):
                    self.pedestrian_crossed = True

                self.prev_mean = current_mean
            else:
                # TODO remove magic number
                rospy.sleep(0.4)
                self.pedestrian_crossed = False
                self.red = 0.0
                self.prev_mean = 0.0
                self.cool_off_start_time = rospy.Time.now().to_sec()

        # TODO remove after time trials
        # if not self.started_timer:
        #     self.score_pub.publish(str("team6,robot,0,XXXX"))
        #     self.started_timer = True

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
        self.vel_pub.publish(self.move)

    def check_crosswalk(self, frame):
        cropped = frame[self.bottom_crop :, :]
        mask = cv2.inRange(cropped, self.lower, self.upper)
        output = cv2.bitwise_and(cropped, cropped, mask=mask)
        self.red = np.mean(output)

    def stop_timer(self, event):
        if not self.stopped_timer:
            self.score_pub.publish(str("team,robot,-1,XXXX"))
            self.stopped_timer = True


if __name__ == "__main__":
    rospy.sleep(1)
    rospy.init_node("CNN_DRIVER")
    it = CNNDriver()
    rospy.spin()
