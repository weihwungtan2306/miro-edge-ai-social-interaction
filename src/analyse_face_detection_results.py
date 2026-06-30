import pandas as pd

def main():
    input_file = "results/face_detection_baseline_log.csv"

    try:
        data = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        print("Run face_detection_logging_test.py first.")
        return

    print("Face Detection Baseline Results")
    print("--------------------------------")
    print(f"Total frames tested: {len(data)}")
    print(f"Average faces detected: {data['faces_detected'].mean():.2f}")
    print(f"Average inference time: {data['inference_time_ms'].mean():.2f} ms")
    print(f"Minimum inference time: {data['inference_time_ms'].min():.2f} ms")
    print(f"Maximum inference time: {data['inference_time_ms'].max():.2f} ms")
    print(f"Average FPS: {data['fps'].mean():.2f}")
    print(f"Minimum FPS: {data['fps'].min():.2f}")
    print(f"Maximum FPS: {data['fps'].max():.2f}")

if __name__ == "__main__":
    main()
    