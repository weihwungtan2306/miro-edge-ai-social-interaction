#!/usr/bin/env python3
import os
import csv
import time
import math
import rospy
from sensor_msgs.msg import CompressedImage, JointState
import numpy as np
from PIL import Image
import io
import miro2 as miro
from pycoral.utils.edgetpu import make_interpreter

MODEL_PATH = "/root/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
CONF_THRESHOLD = 0.5
LOG_EVERY_N = 5
LOG_FILE = "/root/gaze_follow_log.csv"

# Max change in yaw/pitch per frame (radians) -- keeps motion smooth and safe
MAX_STEP_RAD = math.radians(5.0)

interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME", "miro")
kinematic_pub = rospy.Publisher(topic_base_name + "/control/kinematic_joints", JointState, queue_size=0)

# Current commanded state, starts at calibration
current_yaw = miro.constants.YAW_RAD_CALIB
current_pitch = miro.constants.PITCH_RAD_CALIB

frame_count = 0
fps_times = []
last_time = time.time()

log_fh = open(LOG_FILE, "w", newline="")
log_writer = csv.writer(log_fh)
log_writer.writerow([
    "timestamp", "frame", "face_detected", "cx", "cy",
    "target_yaw_deg", "target_pitch_deg",
    "commanded_yaw_deg", "commanded_pitch_deg"
])

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

def step_toward(current, target, max_step):
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + max_step * (1 if delta > 0 else -1)

def publish_kinematic(yaw, pitch):
    joint_cmd = JointState()
    joint_cmd.position = [
        miro.constants.TILT_RAD_CALIB,
        miro.constants.LIFT_RAD_CALIB,
        yaw,
        pitch,
    ]
    kinematic_pub.publish(joint_cmd)

def callback(msg):
    global frame_count, last_time, fps_times, current_yaw, current_pitch

    img = Image.open(io.BytesIO(msg.data)).convert('RGB')
    resized = img.resize((width, height))
    input_data = np.expand_dims(np.array(resized, dtype=np.uint8), axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    faces = [(s, b) for s, b in zip(scores, boxes) if not np.isnan(s) and s >= CONF_THRESHOLD]

    cx = cy = None
    target_yaw_deg = target_pitch_deg = None

    if faces:
        best_score, best_box = max(faces, key=lambda f: f[0])
        ymin, xmin, ymax, xmax = best_box
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        # Convert normalized offset from center into an actual angle using camera FOV
        horizontal_angle = (cx - 0.5) * 2 * miro.constants.CAM_HORI_HALF_FOV
        vertical_angle = (cy - 0.5) * 2 * miro.constants.CAM_VERT_HALF_FOV

        # NOTE: verify sign on first run -- flip if the head turns the wrong way
        target_yaw = clamp(
            miro.constants.YAW_RAD_CALIB - horizontal_angle,
            miro.constants.YAW_RAD_MIN, miro.constants.YAW_RAD_MAX
        )
        target_pitch = clamp(
            miro.constants.PITCH_RAD_CALIB - vertical_angle,
            miro.constants.PITCH_RAD_MIN, miro.constants.PITCH_RAD_MAX
        )

        current_yaw = step_toward(current_yaw, target_yaw, MAX_STEP_RAD)
        current_pitch = step_toward(current_pitch, target_pitch, MAX_STEP_RAD)

        target_yaw_deg = math.degrees(target_yaw)
        target_pitch_deg = math.degrees(target_pitch)

    # If no face: hold last commanded position (do nothing), matches safest behavior
    publish_kinematic(current_yaw, current_pitch)

    now = time.time()
    dt = now - last_time
    last_time = now
    fps_times.append(dt)
    if len(fps_times) > 30:
        fps_times.pop(0)
    avg_fps = len(fps_times) / sum(fps_times) if sum(fps_times) > 0 else 0

    frame_count += 1
    log_writer.writerow([
        now, frame_count, int(bool(faces)),
        cx if cx is not None else "",
        cy if cy is not None else "",
        target_yaw_deg if target_yaw_deg is not None else "",
        target_pitch_deg if target_pitch_deg is not None else "",
        math.degrees(current_yaw), math.degrees(current_pitch),
    ])
    if frame_count % LOG_EVERY_N == 0:
        log_fh.flush()
        status = f"{len(faces)} face(s)" if faces else "no face"
        rospy.loginfo(f"Frame {frame_count}: {status} | yaw={math.degrees(current_yaw):.1f} deg | pitch={math.degrees(current_pitch):.1f} deg | FPS: {avg_fps:.1f}")

def shutdown_hook():
    publish_kinematic(miro.constants.YAW_RAD_CALIB, miro.constants.PITCH_RAD_CALIB)
    log_fh.close()

rospy.init_node('gaze_follow_kinematic', anonymous=True)
rospy.on_shutdown(shutdown_hook)
rospy.Subscriber(topic_base_name + '/sensors/caml/compressed', CompressedImage, callback, queue_size=1, buff_size=2**16)
rospy.loginfo("Gaze-follow (kinematic) node started, listening to " + topic_base_name + "/sensors/caml/compressed...")
rospy.spin()
