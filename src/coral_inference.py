#!/usr/bin/env python3
import time
import numpy as np
import tflite_runtime.interpreter as tflite
import platform
import os

CPU_MODEL   = "/catkin_ws/models/ssd_mobilenet_v2_face_quant_postprocess.tflite"
CORAL_MODEL = "/catkin_ws/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"

def benchmark_cpu(runs=20):
    print("\n--- CPU Benchmark (no accelerator) ---")
    interpreter = tflite.Interpreter(model_path=CPU_MODEL)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    dummy = np.random.randint(0, 255, input_details[0]["shape"], dtype=np.uint8)
    interpreter.set_tensor(input_details[0]["index"], dummy)
    interpreter.invoke()
    times = []
    for i in range(runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        times.append((time.perf_counter() - start) * 1000)
        print(f"  Run {i+1:02d}: {times[-1]:.1f} ms")
    avg = sum(times) / len(times)
    print(f"\n  Average: {avg:.1f} ms | FPS: {1000/avg:.1f} | Min: {min(times):.1f} ms | Max: {max(times):.1f} ms")
    return avg

def benchmark_coral(runs=20):
    print("\n--- Coral Edge TPU Benchmark ---")
    try:
        delegate = tflite.load_delegate("libedgetpu.so.1")
        interpreter = tflite.Interpreter(model_path=CORAL_MODEL, experimental_delegates=[delegate])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        dummy = np.random.randint(0, 255, input_details[0]["shape"], dtype=np.uint8)
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        times = []
        for i in range(runs):
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]["index"], dummy)
            interpreter.invoke()
            times.append((time.perf_counter() - start) * 1000)
            print(f"  Run {i+1:02d}: {times[-1]:.1f} ms")
        avg = sum(times) / len(times)
        print(f"\n  Average: {avg:.1f} ms | FPS: {1000/avg:.1f} | Min: {min(times):.1f} ms | Max: {max(times):.1f} ms")
        return avg
    except Exception as e:
        print(f"  Coral not available: {e}")
        print("  --> Plug in the Coral USB and run again")
        return None

if __name__ == "__main__":
    print("=" * 50)
    print("  MiRo Edge AI - Face Detection Benchmark")
    print(f"  Platform: {platform.machine()}")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    for m in [CPU_MODEL, CORAL_MODEL]:
        status = "OK" if os.path.exists(m) else "MISSING"
        print(f"  [{status}] {os.path.basename(m)}")
    cpu_avg = benchmark_cpu()
    coral_avg = benchmark_coral()
    print("\n" + "=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    print(f"  CPU:   {cpu_avg:.1f} ms  ({1000/cpu_avg:.1f} FPS)")
    if coral_avg:
        speedup = cpu_avg / coral_avg
        under = "YES" if coral_avg < 200 else "NO"
        print(f"  Coral: {coral_avg:.1f} ms  ({1000/coral_avg:.1f} FPS)")
        print(f"  Speedup: {speedup:.1f}x faster on Coral")
        print(f"  Below 200ms HRI threshold: {under}")
    else:
        print("  Coral: NOT CONNECTED - plug in and rerun")
    print("=" * 50)
