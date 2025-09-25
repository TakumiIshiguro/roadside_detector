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
from std_msgs.msg import Int32   # 追加

class roadside_detector:
    def __init__(self):
        rospy.init_node('roadside_detector', anonymous=True)
        self.bridge = CvBridge()

        # --- 新しい Publisher (Int32) ---
        self.class_pub = rospy.Publisher("inter_cls", Int32, queue_size=1)

        self.image_sub = rospy.Subscriber("/camera_center/usb_cam/image_raw", Image, self.callback)
        self.dl = deep_learning()
        self.episode = 0
        self.cv_image = np.zeros((480, 640, 3), np.uint8)
        self.load_path = roslib.packages.get_pkg_dir('roadside_detector') + '/data/model/tsukuba/model.pt'

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        img = resize(self.cv_image, (224, 224), mode='constant')
        prev_pred = 0

        if self.episode == 0:
            self.dl.load(self.load_path)
            print("load model: ", self.load_path)

        pred, conf, probs = self.dl.test(img)
        if conf >= 0.9:
            prev_pred = pred
        else:
            print("-----------------unconfidente------------------")
            pass

        print(f"predicted: {prev_pred}, confidence: {conf:.3f}")

        # --- publish クラスID (0 or 1) ---
        self.class_pub.publish(prev_pred)

        self.episode += 1


if __name__ == '__main__':
    rg = roadside_detector()
    r = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
