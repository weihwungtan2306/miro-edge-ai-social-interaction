import cv2
import time
from ultralytics import YOLO

def main():
    # Load YOLOv8 nano model
    # This will download yolov8n.pt automatically the first time
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("YOLOv8n detection started. Press 'q' to quit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        start_time = time.time()

        # Run YOLO inference
        results = model(frame, verbose=False)

        inference_time = (time.time() - start_time) * 1000

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Draw YOLO results on frame
        annotated_frame = results[0].plot()

        cv2.putText(
            annotated_frame,
            f"Inference: {inference_time:.2f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("YOLOv8n Detection Test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()