#!/usr/bin/env python3

#!/usr/bin/env python3

import os
import rospy

import numpy as np
import cv2, cv_bridge

from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

# cv bridge can't handle compressed img
from sensor_msgs.msg import CompressedImage

class LineFollower(DTROS):
    def __init__(self, node_name, robot_name):
        # initialize the DTROS parent class
        super(LineFollower, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        rospy.Rate(10)

        self.im_pub = rospy.Publisher('/output/image_raw/compressed', CompressedImage, queue_size=1)
        self.imager = rospy.Subscriber(f'/{robot_name}/camera_node/image/compressed', CompressedImage, 
                    self.img_callback, queue_size=1)
        self.mover = rospy.Publisher(f'/{robot_name}/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=10)
        self.low_yellow = np.array([24,150,150], dtype=np.uint8)
        self.high_yellow = np.array([50,255,255], dtype=np.uint8)
        self.low_red = np.array([0 , 120, 120], dtype=np.uint8)
        self.high_red = np.array([10, 255, 255], dtype=np.uint8)

        self.low_yellow = np.array([11,11,64], dtype=np.uint8)
        self.high_yellow = np.array([50,255,255], dtype=np.uint8)
        
        self.yellow_line_centroid = {}
        self.stop_sign_detected = False
        self.yellow_line_detected = False

        self.img_width = 0
        self.decoded_raw_img = None

        # message = Twist2DStamped()
        # message.v = 0
        # message.omega = 0
        # self.mover.publish(message)

        # rospy.on_shutdown(self.shutdown_hook)

    def stop(self):
        # rospy.loginfo("Stop before shutdown")
        message = Twist2DStamped()
        message.v = 0
        message.omega = 0
        self.mover.publish(message)

    def on_shutdown(self):
        rospy.loginfo("Shutdown called")
        for i in range(0, 100):
            self.stop()
        super(LineFollower, self).on_shutdown()
        
    def img_callback(self, ros_data):
        # convert to cv2 friendly format
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.decoded_raw_img = image_np
        # rospy.loginfo("Got here")

        """
        # crop the image
        height, width, channels = image_np.shape
        self.img_width = width
        crop_img_dft = image_np[int(height/2):][0:width]

        # convert to hsv
        hsv_dft = cv2.cvtColor(crop_img_dft, cv2.COLOR_BGR2HSV)

        # filter colors
        mask_yellow = cv2.inRange(hsv_dft, self.low_yellow, self.high_yellow)
        mask_red = cv2.inRange(hsv_dft, self.low_red, self.high_red)

        res_yellow = cv2.bitwise_and(crop_img_dft, crop_img_dft, mask = mask_yellow)
        res_red = cv2.bitwise_and(crop_img_dft, crop_img_dft, mask = mask_red)

        img = CompressedImage()
        img.header.stamp = rospy.Time.now()
        img.format = "jpeg"
        img.data = np.array(cv2.imencode('.jpg', res_yellow)[1]).tostring()

        M_yellow = cv2.moments(mask_yellow, False)
        try:
            self.yellow_line_detected = True
            self.yellow_line_centroid["cx"] = M_yellow["m10"]/M_yellow["m00"]
            self.yellow_line_centroid["cy"] = M_yellow["m01"]/M_yellow["m00"]
            # print("got yellow")
        except ZeroDivisionError:
            self.yellow_line_detected = False
            self.yellow_line_centroid["cx"] = height/2
            self.yellow_line_centroid["cy"] = width/2

        self.im_pub.publish(img)
        """


    def process_img(self):
        if self.decoded_raw_img is not None:
            # crop the image
            height, width, channels = self.decoded_raw_img.shape
            self.img_width = width
            crop_img_dft = self.decoded_raw_img[int(height/2):][0:width]

            # convert to hsv
            hsv_dft = cv2.cvtColor(crop_img_dft, cv2.COLOR_BGR2HSV)

            # filter colors
            mask_yellow = cv2.inRange(hsv_dft, self.low_yellow, self.high_yellow)
            mask_red = cv2.inRange(hsv_dft, self.low_red, self.high_red)

            res_yellow = cv2.bitwise_and(crop_img_dft, crop_img_dft, mask = mask_yellow)
            res_red = cv2.bitwise_and(crop_img_dft, crop_img_dft, mask = mask_red)

            img = CompressedImage()
            img.header.stamp = rospy.Time.now()
            img.format = "jpeg"
            img.data = np.array(cv2.imencode('.jpg', res_yellow)[1]).tostring()

            M_yellow = cv2.moments(mask_yellow, False)
            try:
                self.yellow_line_detected = True
                self.yellow_line_centroid["cx"] = M_yellow["m10"]/M_yellow["m00"]
                self.yellow_line_centroid["cy"] = M_yellow["m01"]/M_yellow["m00"]
                # print("got yellow")
            except ZeroDivisionError:
                self.yellow_line_detected = False
                self.yellow_line_centroid["cx"] = height/2
                self.yellow_line_centroid["cy"] = width/2

            self.im_pub.publish(img)

    def move_based_on_camera(self):
        linear_speed = 0.08
        message = Twist2DStamped()
        message.v = linear_speed
        self.process_img()
        try:
            if self.stop_sign_detected is True:
                # issue stop command
                # stop_cmd = Twist()
                # self.cmd_vel.publish(stop_cmd)
                message.v = 0
                message.omega = 0

            elif self.yellow_line_detected is True:
                cx = self.yellow_line_centroid["cx"]
                cy = self.yellow_line_centroid["cy"]
                error_x = cx - self.img_width/2
                message.omega = -error_x / 25           # was 100 orgiinally
            else:
                message.omega = 0
        except KeyError:
            return

        self.mover.publish(message)

    def run(self):
        rospy.sleep(10)
        rospy.loginfo("Start moving!")
        while not rospy.is_shutdown():
            self.move_based_on_camera()
            # self.process_img()


def main():
    node = LineFollower(node_name="mrrobot22918", robot_name="mrrobot22918")
    node.run()
    rospy.loginfo(f'OpenCV version: {cv2.__version__}')
    rospy.spin()

if __name__ == "__main__":
    main()