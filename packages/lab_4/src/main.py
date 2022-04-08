#!/usr/bin/env python3

#!/usr/bin/env python3

import os
import rospy

import numpy as np
import cv2, cv_bridge

from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

# cv bridge can't handle compressed img


class LineFollower(DTROS):
    def __init__(self, node_name, robot_name, stop_sign_img_path):
        # initialize the DTROS parent class
        super(LineFollower, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        rospy.Rate(10)

        self.im_pub = rospy.Publisher('/output/image_raw/compressed', CompressedImage, queue_size=1)
        self.imager = rospy.Subscriber(f'/{robot_name}/camera_node/image/compressed', CompressedImage, 
                    self.img_callback, queue_size=1)
        self.mover = rospy.Publisher(f'/{robot_name}/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=10)

        # self.low_red = np.array([148, 86, 92], dtype=np.uint8) #51  79
        # self.high_red = np.array([179, 95, 100], dtype=np.uint8) # 255 # 255

        self.low_red = np.array([140, 40, 40], dtype=np.uint8) #51  79 #np.array([148, 86, 92], dtype=np.uint8) #51  79
        self.high_red = np.array([255, 255, 200], dtype=np.uint8) #51  79 #np.array([179, 95, 100], dtype=np.uint8) # 255 # 255

        self.low_stop_red = np.array([11, 11, 64], dtype=np.uint8)
        self.high_stop_red = np.array([50, 255, 255], dtype=np.uint8)

        self.low_yellow = np.array([11,80,80], dtype=np.uint8) # 11 64
        self.high_yellow = np.array([50,255,255], dtype=np.uint8) # 255 255
        
        self.yellow_line_centroid = {}
        self.stop_sign_detected = False
        self.yellow_line_detected = False

        self.img_width = 0
        self.decoded_raw_img = None

        self.bridge = cv_bridge.CvBridge()
        self.red_line_detected = False
        self.AKAZE = cv2.AKAZE_create()
        self.define_stop_sign_input_desc(stop_sign_img_path)

        self.counter = 0

        self.end_program = False

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

    def define_stop_sign_input_desc(self, img_path):
        # reference: https://stackoverflow.com/questions/62581171/how-to-implement-kaze-and-a-kaze-using-python-and-opencv
        stop_img_raw = cv2.imread(img_path)
        hsv = cv2.cvtColor(stop_img_raw, cv2.COLOR_BGR2HSV)
        # rospy.loginfo(f'hsv bud: {hsv}')

        mask_red = cv2.inRange(hsv, self.low_stop_red, self.high_stop_red)

        # # DEBUG
        # img = CompressedImage()
        # img.header.stamp = rospy.Time.now()
        # img.format = "jpeg"
        # img.data = np.array(cv2.imencode('.jpg', mask_red)[1]).tostring()
        
        # while True:
        #     rospy.loginfo("yo")
        #     self.im_pub.publish(img)
            
        rospy.loginfo(f'mask_red: {mask_red}')

        self.stop_base_keypoints, self.stop_base_descriptors = self.AKAZE.detectAndCompute(mask_red, None)
        self.stop_base_descriptors = np.float32(self.stop_base_descriptors)

        rospy.loginfo(f'Init Base Descriptors: {self.stop_base_descriptors}')

        # initialize FLANN algorithm
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        # Create FLANN object
        self.FLANN = cv2.FlannBasedMatcher(indexParams = index_params, searchParams = search_params)

    def match_stop_sign(self, masked_img):
        # reference: https://stackoverflow.com/questions/62581171/how-to-implement-kaze-and-a-kaze-using-python-and-opencv
        candidate_keypoints, candidate_descriptors = self.AKAZE.detectAndCompute(masked_img, None)
        candidate_descriptors = np.float32(candidate_descriptors)

        # rospy.loginfo(f'Base descriptor: {self.stop_base_descriptors}')
        # rospy.loginfo(f'Candidate descriptors: {candidate_descriptors}')

        if candidate_descriptors is None:
            return False

        try:
            matches = self.FLANN.knnMatch(self.stop_base_descriptors, candidate_descriptors, 2)
        except (cv2.error, ValueError):
            return False

        # Lowe's ratio test
        ratio_thresh = 0.7
        good_matches = []

        # filter matches
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        target_matches = 0    # 40
        rospy.loginfo(f'Good matches: {len(good_matches)}')
        if (len(good_matches) > target_matches):
            rospy.loginfo("detected!")
            return True
        return False

    def process_img(self):
        if self.decoded_raw_img is not None:
            # crop the image
            height, width, channels = self.decoded_raw_img.shape
            self.img_width = width
            adjuster = 10
            crop_img_dft = self.decoded_raw_img[int(height/2 + adjuster*9):][0:width]

            # convert to hsv
            hsv_dft = cv2.cvtColor(crop_img_dft, cv2.COLOR_BGR2HSV)

            # filter colors
            mask_yellow = cv2.inRange(hsv_dft, self.low_yellow, self.high_yellow)
            mask_red = cv2.inRange(hsv_dft, self.low_red, self.high_red)

            # try to extract the stop image
            stop_img_crop = self.decoded_raw_img[0:int(height/2)][0:width]
            hsv_stop = cv2.cvtColor(stop_img_crop, cv2.COLOR_BGR2HSV)
            mask_stop = cv2.inRange(hsv_stop, self.low_red, self.high_red)

            ret,thresh = cv2.threshold(mask_yellow, 40, 255, 0)
            contours, _ =cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1) #ccomp

            red_ret,red_thresh = cv2.threshold(mask_stop, 40, 255, 0)
            red_contours, _ =cv2.findContours(red_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1) #ccomp
            

            res_yellow = cv2.bitwise_and(crop_img_dft, crop_img_dft, mask = mask_yellow)

            img = CompressedImage()
            img.header.stamp = rospy.Time.now()
            img.format = "jpeg"

            centres = []
            # go through contours, put centre data on a separate array
            for i in range(len(contours)):
                if (cv2.contourArea(contours[i]) < 1500 ):
                    continue

                m = cv2.moments(contours[i])
                try:
                    centres.append((int(m['m10']/m['m00']), int(m['m01']/m['m00']) ) )
                    cv2.circle(res_yellow, centres[-1], 10, (0,0,255), -1)
                except ZeroDivisionError:
                    pass

            centres.sort(key=lambda cnd: cnd[1]) # order by y coordinate
            
            try:
                self.yellow_line_detected = True
                self.yellow_line_centroid["cx"] = centres[-1][0]
                self.yellow_line_centroid["cy"] = centres[-1][1]
            except Exception as e:
                self.yellow_line_detected = False
                self.yellow_line_centroid["cx"] = height/2
                self.yellow_line_centroid["cy"] = width/2
            

            # M_red = cv2.moments(mask_red)
            # do we see red?
            # rospy.loginfo("Checking red?")
            # if M_red["m00"] > 0 or True:
            if True:
                if self.red_line_detected is False:
                    rospy.loginfo("Red detected")
                    self.red_line_detected = True

                # mask_stop = cv2.inRange(hsv_stop, self.low_red, self.high_red)
                res_stop = cv2.bitwise_and(stop_img_crop, stop_img_crop, mask=mask_stop)

                red_centres = []
                for i in range(len(red_contours)):
                    # cv2.drawContours(res_stop, red_contours, i, (0, 255, 0), 2)
                    
                    approx = cv2.approxPolyDP(red_contours[i], 0.01 * cv2.arcLength(red_contours[i], True), True)
                    if not cv2.isContourConvex(approx):
                        continue

                    # rospy.loginfo(f"{cv2.contourArea(red_contours[i])}")

                    if (cv2.contourArea(red_contours[i]) < 670 ):
                        continue

                    cv2.drawContours(res_stop, red_contours, i, (0, 255, 0), 2)
                    self.stop_sign_detected = True
                    rospy.loginfo(f"OVER 900: {cv2.contourArea(red_contours[i])}")
                    red_centres.append(cv2.contourArea(red_contours[i]))


                # run feature matching
                self.counter+=1
                if self.stop_sign_detected is False and self.counter%3 == 0:
                    rospy.loginfo(len(red_centres) > 0)

                    if (self.match_stop_sign(mask_stop) is True and len(red_centres) > 0 ):
                        self.stop_sign_detected = True
                        rospy.loginfo("Stop sign detected. Stop")

                img.data = np.array(cv2.imencode('.jpg', res_stop)[1]).tostring()
                self.im_pub.publish(img)
                return
            else:
                self.red_line_detected = False

            img.data = np.array(cv2.imencode('.jpg', res_yellow)[1]).tostring()
            # self.im_pub.publish(img)
        else:
            rospy.loginfo("Decoded image is None")

    def move_based_on_camera(self):
        linear_speed = 0.15
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
                self.end_program = True

            elif self.yellow_line_detected is True:
                cx = self.yellow_line_centroid["cx"]
                cy = self.yellow_line_centroid["cy"]
                error_x = cx - self.img_width/2
                message.omega = -error_x / 80 
                rospy.loginfo(-error_x / 80) # 30 for paola
            else:
                message.omega = 0
        except KeyError:
            return

        self.mover.publish(message)

    def run(self):
        rospy.sleep(5)
        rospy.loginfo("Start moving!")
        while (self.end_program is False and not rospy.is_shutdown()):
            self.move_based_on_camera()
            # self.process_img()
            
        rospy.loginfo("Done!")
        # make sure that it stops!
        for i in range(0, 100):
            self.stop()


def main():
    stop_sign_img_path = os.path.dirname(os.path.realpath(__file__)) + "/stop_qr.jpg"

    node = LineFollower(node_name="mrrobot22918", robot_name="mrrobot22918",
                        stop_sign_img_path=stop_sign_img_path)

    # if os.path.isfile(stop_sign_img_path):
    #     rospy.loginfo("Stop image path is VALID")
    # else:
    #     rospy.loginfo("Stop image path is NOT VALID")

    node.run()

    # rospy.loginfo(f'OpenCV version: {cv2.__version__}')
    rospy.spin()

if __name__ == "__main__":
    main()