#!/usr/bin/env python3

import numpy as np
import roslib
roslib.load_manifest('roadside_detector')
import rospy
from cnn import *
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from skimage.transform import resize
import os
import sys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import csv
import time
from sensor_msgs.msg import Joy
import copy
import yaml
from std_srvs.srv import SetBool, SetBoolResponse

class roadside_detector:
    def __init__(self):
        rospy.init_node('roadside_detector', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera_center/usb_cam/image_raw", Image, self.callback)

        self.dl = deep_learning()
        self.action = 0.0
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)
    
        self.inter_flg = False
        self.ignore_flg = False    
        self.learning_flg = False
        self.loop_srv = rospy.Service('/loop_count', SetBool, self.callback_loop_count)
        self.loop_count_flg = False

        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('roadside_detector') + '/data/'
        self.save_dataset_path = self.path + '/dataset/'
        self.save_model_path = self.path + '/model/'

        self.load_dataset_path = self.save_dataset_path + '/tsukuba' + '/dataset.pt'

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

    def callback_loop_count(self, data):
        self.loop_count_flg = data.data

    def joy_callback(self, data):
        if data.buttons[1] == 1:
            self.inter_flg = True
        else: 
            self.inter_flg = False
            
        if data.buttons[2] == 1:
            self.ignore_flg = True
        else:
            self.ignore_flg = False

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            print("No Image")
            return

        img_resized = resize(self.cv_image, (224, 224), mode='constant')
        img_for_train = img_resized.astype(np.float32)  # [224,224,3], 0–1 float32

        # --- 表示用（uint8, 0–255） ---
        img_for_display = (img_resized * 255).astype(np.uint8)  # [224,224,3], 0–255 uint8

        # ラベル付けとデータ作成
        if self.inter_flg:
            self.dl.make_dataset(img_for_train, 1)
            print("label 1")
        elif self.ignore_flg:
            print("-------pass-----")
            pass
        else:
            self.dl.make_dataset(img_for_train, 0)
            print("label 0")

        if self.loop_count_flg:
            img_tensor, label_tensor = self.dl.finalize_dataset() 
            self.dl.save_dataset(self.save_dataset_path, 'dataset.pt')
            self.dl.training()
            self.dl.save(self.save_model_path)
            self.loop_count_flg = False
            os.system('killall roslaunch')
            sys.exit()

        if self.learning_flg:
            self.dl.load_dataset(self.load_dataset_path)
            self.dl.training()
            self.dl.save(self.save_model_path)
            sys.exit()
        else:
            # 表示は BGR に変換して OpenCV ウィンドウに出す
            bgr_img = cv2.cvtColor(img_for_display, cv2.COLOR_RGB2BGR)
            cv2.imshow("resize", bgr_img)
            cv2.waitKey(1)

if __name__ == '__main__':
    rg = roadside_detector()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()