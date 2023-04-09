#!/usr/bin/env python3

################################################################################
## {Description}: Detecting an Apriltag3
## {Description}: Publish /isApriltag topic
## {Description}: If AprilTag3 detected; /isApriltag --> True
## {Description}: If AprilTag3 detected; /isApriltag --> False
################################################################################
## Author: Khairul Izwan Bin Kamsani
## Version: {1}.{0}.{0}
## Email: {wansnap@gmail.com}
################################################################################

"""
Image published (CompressedImage) from tello originally size of 960x720 pixels
We will try to resize it using imutils.resize (with aspect ratio) to width = 320
and then republish it as Image
"""

# import the necessary Python packages

import sys
import cv2
import time
import numpy as np
import imutils
import random
import os

# import the necessary ROS packages
from std_msgs.msg import String, Bool, Float32
from std_msgs.msg import Int16
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import select
if os.name == 'nt':
	import msvcrt
else:
	import tty, termios

import rospy

class CameraAprilTag:
	def __init__(self):

		# OpenCV -- ROS
		self.bridge = CvBridge()
		
		self.twist = Twist()
	
		# state
		self.image_received = False

		
		rospy.logwarn("Line Detection Node [ONLINE]...")
		
		# rospy shutdown
		rospy.on_shutdown(self.cbShutdown)
		self.pub_error = rospy.Publisher('error', Int16, queue_size=10)
		self.pub_angle = rospy.Publisher('angle', Int16, queue_size=10)
		self.pubTakeoff = rospy.Publisher('/tello/takeoff', Empty, queue_size=10)
		# Subscribe to CompressedImage msg
		self.telloImage_topic = "/tello/image_raw/compressed"
		self.telloImage_sub = rospy.Subscriber(
						self.telloImage_topic, 
						CompressedImage, 
						self.cbImage
						)
						
		# Publish to Twist msg
		self.telloTwist_topic = "/tello/cmd_vel"
		self.telloTwist_pub = rospy.Publisher(
					self.telloTwist_topic, 
					Twist, 
					queue_size=10
					)
		self.Kp = 0.012     # Ku=0.14 T=6. PID: p=0.084,i=0.028,d=0.063. PD: p=0.112, d=0.084/1. P: p=0.07
		self.Ki = 0
		self.kd = 1
		self.integral = 0
		self.derivative = 0
		self.last_error = 0
		self.Kp_ang = 0.01        # Ku=0.04 T=2. PID: p=0.024,i=0.024,d=0.006. PD: p=0.032, d=0.008. P: p=0.02/0.01
		self.Ki_ang = 0
		self.kd_ang = 0
		self.integral_ang = 0
		self.derivative_ang = 0
		self.last_ang = 0
		self.was_line = 0
		self.line_side = 0
		self.ctrl_c = False
		self.line_back = 1
		self.landed = 0
		self.takeoffed = 0
		self.error = []
		self.angle = []
		self.fly_time = 0.0
		self.start = 0.0
		self.stop = 0.0
		self.velocity = 0.5 #asal 0.2

		# Allow up to one second to connection
		rospy.sleep(1)

	# Convert image to OpenCV format
	def cbImage(self, msg):

		try:
			# direct conversion to cv2
			self.cv_image = self.bridge.compressed_imgmsg_to_cv2(
								msg, 
								"bgr8"
								)
		except CvBridgeError as e:
			print(e)

		if self.cv_image is not None:
			self.image_received = True
		else:
			self.image_received = False
		
	def cbDetect(self):
		# Info parameters configuration
		fontFace = cv2.FONT_HERSHEY_PLAIN
		fontScale = 0.7
		color = (255, 255, 255)
		colorPose = (0, 0, 255)
		colorIMU = (255, 0, 255)
		thickness = 1
		lineType = cv2.LINE_AA
		bottomLeftOrigin = False # if True (text upside down)
		mask = cv2.inRange(self.cv_image.copy(), (0,0,0), (100,100,100))
		kernel = np.ones((3, 3), np.uint8)
		mask = cv2.erode(mask, kernel, iterations=5)
		mask = cv2.dilate(mask, kernel, iterations=9)
		contours_blk, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours_blk.sort(key=cv2.minAreaRect)

		
		rospy.loginfo(len(contours_blk))

		if len(contours_blk) > 0 and cv2.contourArea(contours_blk[0]) > 5000:
			self.was_line = 1
			blackbox = cv2.minAreaRect(contours_blk[0])
			(x_min, y_min), (w_min, h_min), angle = blackbox
			if angle < -45:
				angle = 90 + angle
			if w_min < h_min and angle > 0:
				angle = (90 - angle) * -1
			if w_min > h_min and angle < 0:
				angle = 90 + angle

			setpoint = self.cv_image.shape[1] / 2
			error = int(x_min - setpoint)
			self.error.append(error)
			self.angle.append(angle)
			normal_error = float(error) / setpoint

			if error > 0:
				self.line_side = 1  # line in right
			elif error <= 0:
				self.line_side = -1  # line in left

			self.integral = float(self.integral + normal_error)
			self.derivative = normal_error - self.last_error
			self.last_error = normal_error


			error_corr = -1 * (self.Kp * normal_error + self.Ki * self.integral + self.kd * self.derivative)  # PID controler
			# print("error_corr:  ", error_corr, "\nP", normal_error * self.Kp, "\nI", self.integral* self.Ki, "\nD", self.kd * self.derivative)

			angle = int(angle)
			normal_ang = float(angle) / 90

			self.integral_ang = float(self.integral_ang + angle)
			self.derivative_ang = angle - self.last_ang
			self.last_ang = angle

			ang_corr = -1 * (self.Kp_ang * angle + self.Ki_ang * self.integral_ang + self.kd_ang * self.derivative_ang)  # PID controler

			box = cv2.boxPoints(blackbox)
			box = np.int0(box)

			cv2.drawContours(self.cv_image, [box], 0, (0, 0, 255), 3)

			cv2.putText(self.cv_image, "Angle: " + str(angle), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)

			cv2.putText(self.cv_image, "Error: " + str(error), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
			cv2.line(self.cv_image, (int(x_min), 200), (int(x_min), 250), (255, 0, 0), 3)


			twist = Twist()
			twist.linear.x = self.velocity
			twist.linear.y = error_corr
			twist.linear.z = 0
			twist.angular.x = 0
			twist.angular.y = 0
			twist.angular.z = ang_corr
			self.telloTwist_pub.publish(twist)
		
			ang = Int16()
			ang.data = angle
			self.pub_angle.publish(ang)

			err = Int16()
			err.data = error
			self.pub_error.publish(err)

			if len(contours_blk) == 0 and self.was_line == 1 and self.line_back == 1:
				twist = Twist()
				if self.line_side == 1:  # line at the right
					twist.linear.y = -0.05 #asal -0.05
					self.telloTwist_pub.publish(twist)
				if self.line_side == -1:  # line at the left
					twist.linear.y = 0.05 #asal 0.05
					self.telloTwist_pub.publish(twist)
			# cv2.imshow("mask", mask)
			cv2.waitKey(1) & 0xFF
	def takeofff(self):
		takeoff = Empty()
		self.pubTakeoff.publish(takeoff)

			
	def distance_to_camera(self, perWidth):
		# compute and return the distance from the maker to the camera
		return (self.knownWidth * self.focalLength) / perWidth

	# Show the output frame
	def cbShowImage(self):

		cv2.imshow("Line Detection", self.cv_image)
#		cv2.imshow("Line Detection Mask", self.thresh)
		cv2.waitKey(1)
		
	# Preview image + info
	def cbPreview(self):
		if(self.takeoffed == 0):
			self.takeofff()
			self.takeoffed =+ 1
		if self.image_received:

			self.cbDetect()
			self.cbShowImage()
		else:
			rospy.logerr("No images recieved")
			
	# rospy shutdown callback
	def cbShutdown(self):
		rospy.logerr("Line Detection Node [OFFLINE]...")
		cv2.destroyAllWindows()

if __name__ == '__main__':
	# Initialize
	rospy.init_node('camera_qr_detection', anonymous=False)
	camera = CameraAprilTag()
	
	r = rospy.Rate(10)
	
	# Camera preview
	while not rospy.is_shutdown():
		camera.cbPreview()
		r.sleep()#!/usr/bin/env python

