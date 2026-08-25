import time
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "ssd_mobilenet_v2_face_quant_postprocess.tflite"
CONF_THRESHOLD = 0.5

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']
print(f"Model input size: {width}x{height}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0). Check camera permissions in Windows Settings > Privacy > Camera.")

frame_times = []
print("Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    orig_h, orig_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height))
    input_data = np.expand_dims(resized.astype(np.uint8), axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    start = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - start) * 1000
    frame_times.append(elapsed_ms)
    if len(frame_times) > 30:
        frame_times.pop(0)

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for box, score in zip(boxes, scores):
        if score >= CONF_THRESHOLD:
            ymin, xmin, ymax, xmax = box
            left, right = int(xmin * orig_w), int(xmax * orig_w)
            top, bottom = int(ymin * orig_h), int(ymax * orig_h)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, f"{score:.2f}", (left, max(0, top - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    avg_ms = sum(frame_times) / len(frame_times)
    fps = 1000 / avg_ms if avg_ms > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}  Latency: {avg_ms:.1f} ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Detection - Baseline (CPU, personal laptop)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if frame_times:
    avg = sum(frame_times) / len(frame_times)
    print(f"\nFinal rolling avg latency: {avg:.2f} ms")
    print(f"Final rolling avg FPS: {1000/avg:.1f}")
