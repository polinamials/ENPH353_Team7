#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import cv2 

import sys
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import time

class LicensePlateDetector():
    def __init__(self):
        self.sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback, queue_size=10)
        self.bridge = CvBridge()

        self.plate_count = 0
        self.lower = np.array([90,0,0])
        self.upper = np.array([130, 10, 10])
        self.cropimgs = []
        self.cropareas = []
        self.gotone = 0
        self.currentplate = []
        
        #self.pub = rospy.Publisher("/license_plate topic")
        #time.sleep(1)
        print("initialized!!!!")
        rospy.spin()

    
    def callback(self,data):
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        query_img = frame[:, int(frame.shape[1]/2):]
        #query_img = frame
        
        mask = cv2.inRange(query_img, self.lower,self.upper)
        output = cv2.bitwise_and(query_img, query_img, mask=mask)
        grey = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(grey, 1, 255,cv2.THRESH_BINARY)

        kernel = np.ones((5,5), np.uint8)
        th = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        dial = cv2.dilate(thresh, kernel, 1)
        contours, hierarchy = cv2.findContours(dial, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas = np.array([cv2.contourArea(c) for c  in contours])

        if len(areas) > 1:
            
            sorted_areas = np.sort(areas)[::-1]
            rect_areas = sorted_areas[0:2]

            blocks = []
            for c in contours:
                if cv2.contourArea(c) in rect_areas:
                    blocks.append(c)
            
            #finds average x value of a contour
            def avg_x(c):
                tot = 0
                for i in range(0,len(c)):
                    tot = tot + c[i][0][0]
                    avg = tot/ len(c)
                return avg

            avg1 = avg_x(blocks[0])
            avg2 = avg_x(blocks[1])

            #assigns which is the right side of the plate and which is the left
            if avg1 > avg2:
                right_block = blocks[0]
                left_block = blocks[1]
            else:
                right_block = blocks[1]
                left_block = blocks[0]


            #makes the contour into a more managable list of points
            def getPoints(c):
                points = []
                for i in range(0,len(c)):
                    point = []
                    point.append(c[i][0][0])
                    point.append(c[i][0][1])
                    points.append(point)
                return points
            left_points = getPoints(left_block)
            right_points = getPoints(right_block)

            # find bottom left point of a set of points
            def find_bottom_left(points):
                x = []
                for p in points:
                    x.append(sum(p))

                min_index = 0
                minval = 100000
                for i in range(0,len(x)):
                    if x[i] < minval:

                        minval = x[i]
                        min_index = i
                return min_index

            # find top right point of a set of points
            def find_top_right(points):
                y = []
                for p in points:
                    y.append(sum(p))
        
                max_index = 0
                maxval = 0
                for j in range(0, len(y)):
                    if y[j] > maxval:
                        maxval = y[j]
                        max_index = j
                return max_index


            
            r_min = right_points[find_bottom_left(right_points)]
            r_max = right_points[find_top_right(right_points)]
            l_min = left_points[find_bottom_left(left_points)]
            l_max = left_points[find_top_right(left_points)]

            #finding top left corner
            l_middle = (l_min[0] + l_max[0]) / 2
            minval= 10000
            index = 0
            for i in range(0,len(left_points)):

                if left_points[i][0] > l_middle:
                    if left_points[i][1] < minval:
                        minval = left_points[i][1]
                        index = i
            top_left_corner = left_points[index]       


            #finding bottom right corner
            r_middle = (r_min[0] + r_max[0]) / 2

            maxval = 0
            kindex = 0
            for j in range(0, len(right_points)):
                if right_points[j][0] < r_middle:
                    if right_points[j][1] > maxval:
                        maxval = right_points[j][1]
                        jindex = j
            bottom_right_corner = right_points[jindex]


            #assigning top right, bottom left
            top_right_corner = r_min
            bottom_left_corner = l_max

            #perspective transform!
            perspective_pts = (top_right_corner, top_left_corner, bottom_left_corner, bottom_right_corner)

            #thanks chatgpt
            def four_point_transform(image, pts):
                """
                Applies a perspective transform to an image given four corner points.

                Args:
                    image (numpy.ndarray): The image to transform.
                    pts (list): The four corner points of the region of interest.

                Returns:
                    numpy.ndarray: The transformed image.
                """
                # Obtain a consistent order of the points and unpack them
                rect = np.zeros((4, 2), dtype = "float32")
                s = np.sum(pts, axis = 1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis = 1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                
                # Compute the width and height of the new image
                (tl, tr, br, bl) = rect
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                
                # Construct the destination points which will be used to obtain a "birds eye view"
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype = "float32")
                
                # Compute the perspective transform matrix and apply it
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
                
                # Return the warped image
                return warped
            
            warp_img = four_point_transform(query_img, perspective_pts) 
            
            # cv2.imshow("image", warp_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            croparea = warp_img.shape[0] * warp_img.shape[1]

            if croparea > 4000:
                self.cropimgs.append(warp_img)
                self.cropareas.append(croparea)
                self.gotone = 1
                print("area over 4k")


        elif len(self.cropimgs) != 0 and self.gotone == 1 :

            maxarea = max(self.cropareas)
            bigindex = self.cropareas.index(maxarea)
            self.currentplate = self.cropimgs[bigindex]
            self.cropareas = []
            self.cropimgs = []
            self.gotone = 0
            print("hey")
            cv2.imshow("image", self.currentplate)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            #self.go = 1
        

        
        # if self.go == 1:

        #     cv2.imshow("image", self.currentplate)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()


            
            
        
       

        
            


        

    

   

        
if __name__ == "__main__":
    rospy.init_node("LICENSE_PLATE_FINDER")
    bestie = LicensePlateDetector()
    
