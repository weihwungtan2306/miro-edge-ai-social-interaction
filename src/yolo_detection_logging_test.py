import cv2
import time
import csv
import os
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    os.makedirs("results", exist_ok=True)
    output_file = "results/yolo_detection_log.csv"

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("YOLOv8n logging started. Press 'q' to quit.")

    prev_time = time.time()
    frame_count = 0

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame",
            "persons_detected",
            "inference_time_ms",
            "fps"
        ])

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            start_time = time.time()
            results = model(frame, verbose=False)
            inference_time = (time.time() - start_time) * 1000

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            frame_count += 1

            person_count = 0

            for box in results[0].boxes:
                class_id = int(box.cls[0])

                # COCO class 0 = person
                if class_id == 0:
                    person_count += 1

            writer.writerow([
                frame_count,
                person_count,
                round(inference_time, 2),
                round(fps, 2)
            ])

            annotated_frame = results[0].plot()

            cv2.putText(
                annotated_frame,
                f"Persons: {person_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                f"Inference: {inference_time:.2f} ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow("YOLOv8n Logging Test", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
    