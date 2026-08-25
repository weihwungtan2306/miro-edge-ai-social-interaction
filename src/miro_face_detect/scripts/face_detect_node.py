#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import time

class FaceDetectNode:
    def __init__(self):
        rospy.init_node('face_detect_node', anonymous=True)
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        self.sub = rospy.Subscriber('/miro/rob01/sensors/caml', Image, self.callback)
        rospy.loginfo("Face detection node started — waiting for MiRo camera frames...")

    def callback(self, msg):
        start = time.perf_counter()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame, verbose=False)
        latency = (time.perf_counter() - start) * 1000
        rospy.loginfo(f"Inference: {latency:.1f}ms | Detections: {len(results[0].boxes)}")

if __name__ == '__main__':
    node = FaceDetectNode()
    rospy.spin()
