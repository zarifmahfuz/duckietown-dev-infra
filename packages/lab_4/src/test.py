#!/usr/bin/env python3

import os
import rospy

import numpy as np
import cv2 


from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

# cv bridge can't handle compressed img
from sensor_msgs.msg import CompressedImage

class MyNode(DTROS):

	# /camera_node/image/compressed

	low_yellow  = np.array([24,150,150], dtype=np.uint8)
	high_yellow = np.array([50,255,255], dtype=np.uint8)

	low_red = np.array( [0 , 120, 120], dtype=np.uint8 )
	high_red = np.array( [10, 255, 255], dtype=np.uint8 )

	red_detected = False

	def __init__(self, node_name):
		# initialize the DTROS parent class
		super(MyNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

		# construct publisher and subscriber
		self.mover = rospy.Publisher('/csc22913/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=10)
		self.im_pub = rospy.Publisher('/output/image_raw/compressed', CompressedImage, queue_size=1)
		self.imager = rospy.Subscriber("/csc22913/camera_node/image/compressed", CompressedImage, self.img_callback,  queue_size = 1)

		rospy.on_shutdown(self.shutdown_hook)

	def shutdown_hook(self):
		message.v = 0
		message.omega = 0
		self.mover.publish(message)

	def img_callback(self, ros_data):
		nothing = False
		is_red = False
		# convert to cv2 friendly format
		np_arr = np.fromstring(ros_data.data, np.uint8)
		image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

		# crop the image
		adjuster = 100
		height, width, channels = image_np.shape

		# if self.red_detected:
		# 	# crop_img_dft = image_np[1:height-adjuster*2][adjuster*3:width-adjuster*3]
		# 	crop_img_dft = image_np[1:height-adjuster*3][adjuster:width-adjuster]
		# 	print("top heigh: ", height-adjuster*3)
		# 	print("left:", adjuster*3)
		# 	print("right:", width-adjuster*3)
		# else:
		# crop_img_dft = image_np[adjuster*3 - 20 :width-adjuster*3 + 20][0:height]

		crop_img_dft = image_np[int(height/2):][0:width]

		# CONVERT TO HSV
		hsv_dft = cv2.cvtColor(crop_img_dft, cv2.COLOR_BGR2HSV)

		# filter colors
		mask_yellow = cv2.inRange(hsv_dft, self.low_yellow, self.high_yellow)
		mask_red = cv2.inRange(hsv_dft, self.low_red, self.high_red)

		#   # see only the sign
		#   sign_detect = cv2.inRange(hsv_high, low_red, high_red)

		res_yellow = cv2.bitwise_and(crop_img_dft, crop_img_dft, mask = mask_yellow)
		res_red = cv2.bitwise_and(crop_img_dft, crop_img_dft, mask = mask_red)

		yellow = cv2.cvtColor(res_yellow, cv2.COLOR_BGR2GRAY)
		red = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)

		message = Twist2DStamped()
		img = CompressedImage()
		img.header.stamp = rospy.Time.now()
		img.format = "jpeg"

		# if np.sum(red) > 400000:
		# 	chosen_mask = mask_red
		# 	res = res_red
		# 	# nothing = True
		# 	self.red_detected = True
		# 	print("Det: RED")
		# 	message.v = 0.15
		# 	img.data = np.array(cv2.imencode('.jpg', res_red)[1]).tostring()
		# else:
		chosen_mask = mask_yellow
		res = res_yellow
		self.red_detected = False # no longer detected red
		message.v = 0.15
		# print("Det: YELL")

		img.data = np.array(cv2.imencode('.jpg', res_yellow)[1]).tostring()


		# get the moment
		m = cv2.moments(chosen_mask, False)
		try:
		    cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
		    # print("got yellow")
		except ZeroDivisionError:
		    cx, cy = height/2, width/2
		    nothing = True
		    print("Det: NO")

		if not nothing:
			error_x = cx - width/2
			message.omega = -error_x/25 # was 100 orgiinally
			if (self.red_detected):
				print("red amount:", np.sum(red))

				print("cx:", cx)

				# if cx > 260 and cs < 350:
				# 	error_x = cx - width/2
				# 	message.omega = -error_x/25
				# else:

				# 	if cx < 260:
				# 		message.omega = -3
				# 	elif cx >300:
				# 		message.omega = 3

			print("Dir: ANGLED")
		else:
			message.omega = 0
			print("Dir: STRAIGHT")

		self.im_pub.publish(img)
		self.mover.publish(message)


	def run(self):
		pass
		# # publish message every 1 second
		# rate = rospy.Rate(1) # 1Hz
		# while not rospy.is_shutdown():

			# left turning:
			# left motor: -0.5
			# right motor: 0.2
			# omega +8?

			# right turning: note that the r motor is always a little weaker on my duckie
			# l motor: 0.4
			# r motor: -0.3
			# omega -8


			# message = Twist2DStamped()
			# message.v = 0.2
			# message.omega =4
			# print("4")

			# # 0.3 left slight rotation
			# self.mover.publish(message)
			# rate.sleep()

			# message.v = 0
			# message.omega = 0
			# self.mover.publish(message)
			# rate.sleep()

			# message.v = 0.2
			# message.omega = -4
			# self.mover.publish(message)
			# print("-4")
			# rate.sleep()

			# message.v = 0
			# message.omega = 0
			# self.mover.publish(message)
			# rate.sleep()



# if __name__ == '__main__':
#     # create the node
#     node = MyPublisherNode(node_name='csc22913_gp')
#     # run node
#     node.run()
#     # keep spinning
#     rospy.spin()





# DEBUG_FLAG = True
# DFLT_SPEED = 0.2

# # IMAGE PROCESSING AND FEATURE MATCHING
# BRIDGE = CvBridge()
# AKAZE = cv2.AKAZE_create()
# FLANN_INDEX_KDTREE = 1

# #FILTERS
# # WHITE
# low_white  = np.array([0,0,195], dtype=np.uint8)
# high_white = np.array([0,0,255], dtype=np.uint8)

# # BLUE
# low_blue  = np.array([100,0,200], dtype=np.uint8)
# high_blue = np.array([255,255,255], dtype=np.uint8)

# # GREEN
# low_green  = np.array([50,70,200], dtype=np.uint8)
# high_green = np.array([100,255,255], dtype=np.uint8) 

# # RED
# low_red = np.array( [0 , 50, 50] )
# high_red = np.array( [10, 255, 255] )

# # GLOBAL variables
# redCnt = 0

# #
# # Gathers necessary keypoints and descriptors for a stop sign
# # This is the data we will be comparing against for matching.
# # REFERENCED:
# # https://stackoverflow.com/questions/62581171/how-to-implement-kaze-and-a-kaze-using-python-and-opencv
# #
# def init_original_comparator():
#   global descriptors2
#   global FLANN
#   img2_raw = cv2.imread("closeup.png")
#   hsv_img2 = cv2.cvtColor(img2_raw, cv2.COLOR_BGR2HSV)
#   img2 = cv2.inRange(hsv_img2, low_red, high_red) # filters only the red

#   keypoints2, descriptors2 = AKAZE.detectAndCompute(img2, None)

#   # set up FLANN
#   index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#   search_params = dict(checks = 50)
#   FLANN = cv2.FlannBasedMatcher(indexParams = index_params,
#                    searchParams = search_params)

#   descriptors2 = np.float32(descriptors2)



# # 
# # Checks if a match for the stop sign has been found.
# # REFERENCED: 
# # https://stackoverflow.com/questions/62581171/how-to-implement-kaze-and-a-kaze-using-python-and-opencv
# #
# def feature_match_result(img1):
#   global descriptors2
#   global FLANN
#   if (img1 is None):
#     print('BAD IMAGE- ABORT')
#     exit(0)

#   #-- Step 1: Detect the keypoints using AKAZE Detector, compute the descriptors
#   keypoints1, descriptors1 = AKAZE.detectAndCompute(img1, None)

#   if (DEBUG_FLAG):
#     print ("Descriptor1, data type: ",type(descriptors1))

#   # Convert to float32
#   descriptors1 = np.float32(descriptors1)

#   if (DEBUG_FLAG):
#     print ("Descriptor1, data type: ",type(descriptors1))

#   if (descriptors1 is not None):
#     if (DEBUG_FLAG):
#       print ("VALID descriptors1")

#     # Matching descriptor vectors using FLANN Matcher
#     matches = FLANN.knnMatch(queryDescriptors = descriptors1,
#                     trainDescriptors = descriptors2,
#                     k = 2)

#     # Lowe's ratio test
#     ratio_thresh = 0.7

#     # "Good" matches
#     good_matches = []

#     # Filter matches
#     for m, n in matches:
#         if m.distance < ratio_thresh * n.distance:
#             good_matches.append(m)
	
#     if (len(good_matches) > 19):
#       print("MATCH!")
#       return 1

#   print("NOT A MATCH")
#   return 0

def main():
	node = MyNode(node_name='csc22913_line_follower')
	# run node
	# init_original_comparator()
	node.run()

	# keep spinning
	rospy.spin()

if __name__ == "__main__":
  main()