#!/usr/bin/env python3
"""
MiRo Edge AI - Full Pipeline Benchmark
Face Detection + Emotion Recognition + Speech-to-Text
Measures combined latency for social interaction pipeline
"""

import time
import numpy as np
import cv2
import whisper
import tflite_runtime.interpreter as tflite
from deepface import DeepFace

print("=" * 55)
print("  MiRo Edge AI - Full Pipeline Benchmark")
print("  Face Detection + Emotion + Speech-to-Text")
print("=" * 55)

FACE_MODEL = "/catkin_ws/models/ssd_mobilenet_v2_face_quant_postprocess.tflite"

print("\nLoading models...")
face_interpreter = tflite.Interpreter(model_path=FACE_MODEL)
face_interpreter.allocate_tensors()
face_input = face_interpreter.get_input_details()
print("  [OK] Face detection model loaded")

whisper_model = whisper.load_model("tiny.en")
print("  [OK] Whisper tiny.en loaded")
print("  [OK] DeepFace ready (loads on first call)")

dummy_frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
dummy_audio = np.zeros(16000 * 3, dtype=np.float32)

print("\nWarming up...")
face_interpreter.set_tensor(face_input[0]["index"],
    np.random.randint(0, 255, face_input[0]["shape"], dtype=np.uint8))
face_interpreter.invoke()
whisper_model.transcribe(dummy_audio, language="en", fp16=False)
try:
    DeepFace.analyze(dummy_frame, actions=["emotion"],
        enforce_detection=False, silent=True)
except:
    pass
print("  Warmup complete")

print("\nRunning full pipeline benchmark (10 runs)...")
pipeline_times = []
face_times = []
emotion_times = []
stt_times = []

for i in range(10):
    total_start = time.perf_counter()

    t1 = time.perf_counter()
    face_interpreter.set_tensor(face_input[0]["index"],
        np.random.randint(0, 255, face_input[0]["shape"], dtype=np.uint8))
    face_interpreter.invoke()
    face_times.append((time.perf_counter() - t1) * 1000)

    t2 = time.perf_counter()
    try:
        DeepFace.analyze(dummy_frame, actions=["emotion"],
            enforce_detection=False, silent=True)
    except:
        pass
    emotion_times.append((time.perf_counter() - t2) * 1000)

    t3 = time.perf_counter()
    whisper_model.transcribe(dummy_audio, language="en", fp16=False)
    stt_times.append((time.perf_counter() - t3) * 1000)

    total = (time.perf_counter() - total_start) * 1000
    pipeline_times.append(total)
    print(f"  Run {i+1:02d}: {total:.0f}ms total "
          f"(face:{face_times[-1]:.0f}ms "
          f"emotion:{emotion_times[-1]:.0f}ms "
          f"stt:{stt_times[-1]:.0f}ms)")

print(f"\n{'=' * 55}")
print("  FULL PIPELINE RESULTS SUMMARY")
print(f"{'=' * 55}")
print(f"  Face detection:     {sum(face_times)/len(face_times):.0f}ms avg")
print(f"  Emotion recognition:{sum(emotion_times)/len(emotion_times):.0f}ms avg")
print(f"  Speech-to-text:     {sum(stt_times)/len(stt_times):.0f}ms avg")
print(f"  TOTAL PIPELINE:     {sum(pipeline_times)/len(pipeline_times):.0f}ms avg")
print(f"  Below 200ms threshold: {'YES' if sum(pipeline_times)/len(pipeline_times) < 200 else 'NO - edge acceleration needed'}")
print(f"{'=' * 55}")
