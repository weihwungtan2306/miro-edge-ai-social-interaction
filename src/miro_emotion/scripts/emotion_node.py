#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import urllib.request
import os

class EmotionNode:
    def __init__(self):
        rospy.init_node('emotion_node', anonymous=True)
        self.bridge = CvBridge()
        self.emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
        self.net = self.load_model()
        self.pub = rospy.Publisher('/miro/emotion', String, queue_size=10)
        self.sub = rospy.Subscriber('/miro/rob01/sensors/caml', Image, self.callback)
        rospy.loginfo("Emotion recognition node started")

    def load_model(self):
        model_path = '/catkin_ws/models/emotion_model.caffemodel'
        proto_path = '/catkin_ws/models/emotion_model.prototxt'
        if not os.path.exists(model_path):
            rospy.loginfo("Downloading emotion model...")
            urllib.request.urlretrieve(
                'https://github.com/oarriaga/face_classification/raw/master/trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5',
                '/catkin_ws/models/emotion_raw.h5'
            )
        net = cv2.dnn.readNetFromONNX('/catkin_ws/models/emotion.onnx') if os.path.exists('/catkin_ws/models/emotion.onnx') else None
        return net

    def preprocess_face(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        return normalized.reshape(1, 1, 48, 48).astype(np.float32)

    def callback(self, msg):
        start = time.perf_counter()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        face_region = cv2.resize(frame, (48, 48))
        latency = (time.perf_counter() - start) * 1000
        rospy.loginfo(f"Emotion preprocessing: {latency:.1f}ms")

if __name__ == '__main__':
    node = EmotionNode()
    rospy.spin()
