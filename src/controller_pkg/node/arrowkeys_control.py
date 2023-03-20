#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import keyboard

rospy.init_node("topic_publisher")
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

rate = rospy.Rate(2)
move = Twist()
move.linear.x = 0.0
move.angular.z = 0.0

while not rospy.is_shutdown():
    if keyboard.is_pressed("w"):
        move.linear.x = 0.5

    if keyboard.is_pressed("a"):
        move.linear.x = 0.0
        move.angular.z = -0.0

    if keyboard.is_pressed("s"):
        move.angular.z = -0.5

    if keyboard.is_pressed("d"):
        move.angular.z = 0.5

    pub.publish(move)
    rate.sleep()
