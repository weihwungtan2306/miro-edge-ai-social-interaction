#!/usr/bin/env python3
"""
MiRo Edge AI - Full Social Interaction Pipeline ROS Node
Connects face detection + emotion recognition + speech-to-text
Subscribes to MiRo camera and microphone topics
Publishes emotion and speech results as ROS topics
"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading
import tflite_runtime.interpreter as tflite
from deepface import DeepFace
import whisper

class MiRoSocialPipeline:
    def __init__(self):
        rospy.init_node("miro_social_pipeline", anonymous=True)
        rospy.loginfo("MiRo Social Pipeline starting...")

        self.bridge = CvBridge()
        self.face_model = "/catkin_ws/models/ssd_mobilenet_v2_face_quant_postprocess.tflite"

        rospy.loginfo("Loading face detection model...")
        self.face_interpreter = tflite.Interpreter(model_path=self.face_model)
        self.face_interpreter.allocate_tensors()
        self.face_input  = self.face_interpreter.get_input_details()
        self.face_output = self.face_interpreter.get_output_details()
        rospy.loginfo("  [OK] Face detection loaded")

        rospy.loginfo("Loading Whisper STT model...")
        self.whisper_model = whisper.load_model("tiny.en")
        rospy.loginfo("  [OK] Whisper tiny.en loaded")

        rospy.loginfo("  [OK] DeepFace emotion ready (loads on first call)")

        self.emotion_pub  = rospy.Publisher("/miro/social/emotion", String, queue_size=10)
        self.face_pub     = rospy.Publisher("/miro/social/face_detected", String, queue_size=10)
        self.speech_pub   = rospy.Publisher("/miro/social/speech", String, queue_size=10)
        self.latency_pub  = rospy.Publisher("/miro/social/pipeline_latency", Float32, queue_size=10)

        self.cam_sub = rospy.Subscriber(
            "/miro/rob01/sensors/caml", Image, self.camera_callback)
        rospy.loginfo("  [OK] Subscribed to /miro/rob01/sensors/caml")

        self.current_emotion  = "unknown"
        self.current_face     = "none"
        self.face_confidence  = 0.0
        self.pipeline_latency = 0.0
        self.frame_count      = 0
        self.emotion_lock     = threading.Lock()
        self.analyzing        = False

        rospy.loginfo("MiRo Social Pipeline ready!")
        rospy.loginfo("Publishing to:")
        rospy.loginfo("  /miro/social/emotion")
        rospy.loginfo("  /miro/social/face_detected")
        rospy.loginfo("  /miro/social/speech")
        rospy.loginfo("  /miro/social/pipeline_latency")

    def run_face_detection(self, frame):
        resized = cv2.resize(frame, (320, 320))
        input_data = np.expand_dims(resized, axis=0)
        self.face_interpreter.set_tensor(self.face_input[0]["index"], input_data)
        self.face_interpreter.invoke()
        scores = self.face_interpreter.get_tensor(self.face_output[2]["index"])
        best_score = float(np.max(scores))
        return best_score

    def run_emotion(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=["emotion"],
                enforce_detection=False, silent=True)
            emotion = result[0]["dominant_emotion"]
            score   = result[0]["emotion"][emotion]
            return emotion, score
        except:
            return "unknown", 0.0

    def camera_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % 5 != 0:
            return
        if self.analyzing:
            return

        self.analyzing = True
        def process():
            try:
                pipeline_start = time.perf_counter()
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

                t1 = time.perf_counter()
                face_score = self.run_face_detection(frame)
                face_time  = (time.perf_counter() - t1) * 1000

                if face_score > 0.3:
                    self.current_face    = "detected"
                    self.face_confidence = face_score
                    self.face_pub.publish(f"detected:{face_score:.3f}")

                    t2 = time.perf_counter()
                    emotion, emo_score = self.run_emotion(frame)
                    emotion_time = (time.perf_counter() - t2) * 1000
                    self.current_emotion = emotion
                    self.emotion_pub.publish(f"{emotion}:{emo_score:.1f}")
                    rospy.loginfo(f"Face: {face_score:.3f} | Emotion: {emotion} ({emo_score:.1f}%) | Face:{face_time:.0f}ms Emotion:{emotion_time:.0f}ms")
                else:
                    self.current_face = "none"
                    self.face_pub.publish("none:0.000")
                    rospy.loginfo(f"No face detected (score: {face_score:.3f}) | Latency: {face_time:.0f}ms")

                total_latency = (time.perf_counter() - pipeline_start) * 1000
                self.pipeline_latency = total_latency
                self.latency_pub.publish(total_latency)
            finally:
                self.analyzing = False

        t = threading.Thread(target=process, daemon=True)
        t.start()

if __name__ == "__main__":
    try:
        node = MiRoSocialPipeline()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
