#!/usr/bin/env python3

import os
import numpy as np
import yaml
import cv2
import os
from tag import Tag
from pupil_apriltags import Detector
from duckietown.dtros import DTROS, NodeType


class MyNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
     
        # TODO: add your subsribers or publishers here

        # TODO: add information about tags
        TAG_SIZE = .08
        FAMILIES = "tagStandard41h12"
        self.tags = Tag(TAG_SIZE, FAMILIES)

        # Add information about tag locations
        # Function Arguments are id, x, y, z, theta_x, theta_y, theta_z (euler) 
        # for example, self.tags.add_tag( ... 


        # Load camera parameters
        # TODO: change with your robots name
        with open("/data/config/calibrations/camera_intrinsic/MYROBOT.yaml") as file:
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

