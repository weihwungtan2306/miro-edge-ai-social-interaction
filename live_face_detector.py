#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np
from PIL import Image, ImageDraw
import io
import time
from pycoral.utils.edgetpu import make_interpreter

MODEL_PATH = "/root/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
CONF_THRESHOLD = 0.5
SAVE_EVERY_N = 30

interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

frame_count = 0
fps_times = []
last_time = time.time()

def callback(msg):
    global frame_count, last_time, fps_times
    img = Image.open(io.BytesIO(msg.data)).convert('RGB')
    orig_w, orig_h = img.size
    resized = img.resize((width, height))
    input_data = np.expand_dims(np.array(resized, dtype=np.uint8), axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    faces = [(s, b) for s, b in zip(scores, boxes) if not np.isnan(s) and s >= CONF_THRESHOLD]

    now = time.time()
    dt = now - last_time
    last_time = now
    fps_times.append(dt)
    if len(fps_times) > 30:
        fps_times.pop(0)
    avg_fps = len(fps_times) / sum(fps_times) if sum(fps_times) > 0 else 0

    frame_count += 1
    rospy.loginfo(f"Frame {frame_count}: {len(faces)} face(s) | pipeline FPS: {avg_fps:.1f}")

    if frame_count % SAVE_EVERY_N == 0:
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        for score, box in faces:
            ymin, xmin, ymax, xmax = box
            left, right = xmin * orig_w, xmax * orig_w
            top, bottom = ymin * orig_h, ymax * orig_h
            draw.rectangle([left, top, right, bottom], outline="red", width=4)
            draw.text((left, max(0, top - 12)), f"{score:.2f}", fill="red")
        out_path = f"/root/live_frame_{frame_count}.jpg"
        draw_img.save(out_path)
        rospy.loginfo(f"Saved snapshot: {out_path}")

rospy.init_node('coral_face_detector', anonymous=True)
rospy.Subscriber('/miro/sensors/caml/compressed', CompressedImage, callback, queue_size=1, buff_size=2**24)
rospy.loginfo("Coral face detector node started, listening to /miro/sensors/caml/compressed...")
rospy.spin()
