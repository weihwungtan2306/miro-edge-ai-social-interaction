import cv2
import time
import csv
import os

def main():
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        print("Error: Could not load face detector.")
        return

    os.makedirs("results", exist_ok=True)

    output_file = "results/face_detection_baseline_log.csv"

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Face detection logging started. Press 'q' to quit.")

    prev_time = time.time()
    frame_count = 0

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame",
            "faces_detected",
            "inference_time_ms",
            "fps"
        ])

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            start_time = time.time()

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )

            inference_time = (time.time() - start_time) * 1000

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            frame_count += 1

            writer.writerow([
                frame_count,
                len(faces),
                round(inference_time, 2),
                round(fps, 2)
            ])

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Inference: {inference_time:.2f} ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Face Detection Logging Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
    