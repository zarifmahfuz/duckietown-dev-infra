#!/usr/bin/env python3

import os
import numpy as np
import rospy
import yaml
import cv2, cv_bridge
import os
import math
from tag import Tag
# from pupil_apriltags import Detector
from dt_apriltags import Detector
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage


class MyNode(DTROS):

    def __init__(self, node_name, robot_name):
        # initialize the DTROS parent class
        super(MyNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
     
        # TODO: add your subsribers or publishers here
        self.decoded_raw_img = None
        self.undistorted_image = None
        self.imager = rospy.Subscriber(f'/{robot_name}/camera_node/image/compressed', CompressedImage, 
                      self.img_callback, queue_size=1)
        self.im_pub = rospy.Publisher('/output/image_raw/compressed', CompressedImage, queue_size=1)

        # TODO: add information about tags
        TAG_SIZE = .08
        FAMILIES = "tagStandard41h12"
        self.tags = Tag(TAG_SIZE, FAMILIES)

        # Add information about tag locations
        # Function Arguments are id, x, y, z, theta_x, theta_y, theta_z (euler) 
        # for example, self.tags.add_tag( ... 
        self.tags.add_tag(id=0, x=0, y=0, z=0.3048, theta_x=0, theta_y=0, theta_z=0)
        self.tags.add_tag(id=1, x=0.3048, y=0, z=0.6096, theta_x=0, theta_y=-math.pi / 2, theta_z=0)
        self.tags.add_tag(id=2, x=0.6096, y=0, z=0.3048, theta_x=0, theta_y=-math.pi, theta_z=0)
        self.tags.add_tag(id=3, x=0.3048, y=0, z=0, theta_x=0, theta_y=math.pi / 2, theta_z=0)


        # Load camera parameters
        # TODO: change with your robots name
        with open(f'/data/config/calibrations/camera_intrinsic/{robot_name}.yaml') as file:
                camera_list = yaml.load(file,Loader = yaml.FullLoader)

        self.camera_intrinsic_matrix = np.array(camera_list['camera_matrix']['data']).reshape(3,3)
        self.distortion_coeff = np.array(camera_list['distortion_coefficients']['data']).reshape(5,1)


    def undistort(self, img):
        '''
        Takes a fisheye-distorted image and undistorts it

        Adapted from: https://github.com/asvath/SLAMDuck
        '''
        height = img.shape[0]
        width = img.shape[1]

        newmatrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_intrinsic_matrix,
            self.distortion_coeff, 
            (width, height),
            1, 
            (width, height))

        map_x, map_y = cv2.initUndistortRectifyMap(
            self.camera_intrinsic_matrix, 
            self.distortion_coeff,  
            np.eye(3), 
            newmatrix, 
            (width, height), 
            cv2.CV_16SC2)

        undistorted_image = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
       
        return undistorted_image   
             

    def detect(self, img):
        '''
        Takes an images and detects AprilTags
        '''
        PARAMS = [
            self.camera_intrinsic_matrix[0,0],
            self.camera_intrinsic_matrix[1,1],
            self.camera_intrinsic_matrix[0,2],
            self.camera_intrinsic_matrix[1,2]] 


        TAG_SIZE = 0.08 
        detector = Detector(families="tagStandard41h12", nthreads=1)
        detected_tags = detector.detect(
            img, 
            estimate_tag_pose=True, 
            camera_params=PARAMS, 
            tag_size=TAG_SIZE)

        return detected_tags

    def img_callback(self, ros_data):
        # convert to cv2 friendly format
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.decoded_raw_img = image_np
        self.undistorted_image = self.undistort(image_np)

    def publish_img(self, img):
        cmp_img = CompressedImage()
        cmp_img.header.stamp = rospy.Time.now()
        cmp_img.format = "jpeg"
        cmp_img.data = np.array(cv2.imencode('.jpg', img)[1]).tostring()
        self.im_pub.publish(cmp_img)

    def run(self):
        while True:
            rospy.loginfo("Entered loop")
            if self.undistorted_image is None:
                continue
            self.publish_img(self.undistorted_image)
            detected_tags = self.detect(self.undistorted_image)
            rospy.loginfo(f'Detected tags: {detected_tags}')
            rospy.sleep(50)


def main():
    node = MyNode(node_name="mrrobot22918", robot_name="mrrobot22918")
    node.run()
    rospy.spin()


if __name__ == "__main__":
    main()
