#!/usr/bin/env python3
"""
Live Whisper Speech-to-Text Node
Publishes transcript to /miro/social/speech ROS topic
"""

import rospy
from std_msgs.msg import String
import whisper
import numpy as np
import time
import threading
import queue

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False
    print("Audio not available - running in simulation mode")

SAMPLE_RATE = 16000
CHUNK_DURATION = 3
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION

class WhisperSTTNode:
    def __init__(self):
        rospy.init_node("whisper_stt_node", anonymous=True)
        rospy.loginfo("Whisper STT Node starting...")

        rospy.loginfo("Loading Whisper tiny.en model...")
        self.model = whisper.load_model("tiny.en")
        rospy.loginfo("  [OK] Whisper tiny.en loaded")

        self.speech_pub = rospy.Publisher("/miro/social/speech", String, queue_size=10)
        self.audio_queue = queue.Queue()
        self.running = True

        rospy.loginfo("Whisper STT Node ready!")
        rospy.loginfo("Publishing to: /miro/social/speech")

    def transcribe_chunk(self, audio_data):
        start = time.perf_counter()
        audio_float = audio_data.astype(np.float32) / 32768.0
        result = self.model.transcribe(audio_float, language="en", fp16=False)
        latency = (time.perf_counter() - start) * 1000
        transcript = result["text"].strip()
        return transcript, latency

    def run_simulation(self):
        rospy.loginfo("Running in simulation mode (no microphone)")
        phrases = [
            "Hello MiRo how are you today",
            "Can you come here please",
            "I am feeling happy today",
            "What can you do",
            "Good morning MiRo"
        ]
        idx = 0
        while not rospy.is_shutdown():
            dummy_audio = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            transcript, latency = self.transcribe_chunk(
                (dummy_audio * 32768).astype(np.int16))
            simulated = phrases[idx % len(phrases)]
            rospy.loginfo(f"[SIM] Transcript: '{simulated}' | Latency: {latency:.0f}ms")
            self.speech_pub.publish(simulated)
            idx += 1
            rospy.sleep(4)

    def audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(indata.copy())

    def run_live(self):
        rospy.loginfo("Recording from microphone...")
        rospy.loginfo("Speak into your microphone - transcription will appear below")
        buffer = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        pos = 0
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                           dtype=np.float32, blocksize=1024,
                           callback=self.audio_callback):
            while not rospy.is_shutdown():
                try:
                    chunk = self.audio_queue.get(timeout=1.0)
                    chunk_flat = chunk.flatten()
                    space = CHUNK_SAMPLES - pos
                    if len(chunk_flat) >= space:
                        buffer[pos:] = chunk_flat[:space]
                        audio_int16 = (buffer * 32768).astype(np.int16)
                        transcript, latency = self.transcribe_chunk(audio_int16)
                        if transcript:
                            rospy.loginfo(f"Transcript: '{transcript}' | Latency: {latency:.0f}ms")
                            self.speech_pub.publish(transcript)
                        buffer = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
                        pos = 0
                    else:
                        buffer[pos:pos+len(chunk_flat)] = chunk_flat
                        pos += len(chunk_flat)
                except queue.Empty:
                    continue

    def run(self):
        if AUDIO_AVAILABLE:
            self.run_live()
        else:
            self.run_simulation()

if __name__ == "__main__":
    try:
        node = WhisperSTTNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
