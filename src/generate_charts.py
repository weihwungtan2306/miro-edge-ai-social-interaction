#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

print("Generating dissertation result charts...")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("MiRo Edge AI — CPU Baseline Results\nUniversity of Sheffield Dissertation 2026",
             fontsize=14, fontweight="bold", y=1.02)

models = ["YOLOv8n\n(dummy)", "SSD MobileNetV2\n(dummy)", "SSD MobileNetV2\n(real face)"]
latencies = [59.5, 166.6, 169.0]
colors = ["#2E75B6", "#2E75B6", "#1F4E79"]
ax1 = axes[0]
bars = ax1.bar(models, latencies, color=colors, width=0.5, edgecolor="white")
ax1.axhline(y=200, color="red", linestyle="--", linewidth=2, label="200ms HRI threshold")
ax1.set_title("Face Detection Latency\n(CPU only)", fontweight="bold")
ax1.set_ylabel("Latency (ms)")
ax1.set_ylim(0, 300)
ax1.legend()
for bar, val in zip(bars, latencies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"{val}ms", ha="center", va="bottom", fontweight="bold", fontsize=10)

components = ["Face\nDetection", "Emotion\nRecognition", "Speech-to-Text"]
dummy_times = [166.6, 31.0, 228.0]
real_times = [169.0, 1005.2, 546.9]
x = np.arange(len(components))
width = 0.35
ax2 = axes[1]
b1 = ax2.bar(x - width/2, dummy_times, width, label="Dummy input", color="#2E75B6", edgecolor="white")
b2 = ax2.bar(x + width/2, real_times, width, label="Real face/audio", color="#1F4E79", edgecolor="white")
ax2.axhline(y=200, color="red", linestyle="--", linewidth=2, label="200ms threshold")
ax2.set_title("Pipeline Component Comparison\n(dummy vs real input)", fontweight="bold")
ax2.set_ylabel("Latency (ms)")
ax2.set_xticks(x)
ax2.set_xticklabels(components)
ax2.legend()
ax2.set_ylim(0, 1200)

pipeline_labels = ["Dummy\nInput Pipeline", "Real Face\nPipeline"]
pipeline_totals = [424.0, 1721.1]
colors2 = ["#2E75B6", "#1F4E79"]
ax3 = axes[2]
bars3 = ax3.bar(pipeline_labels, pipeline_totals, color=colors2, width=0.4, edgecolor="white")
ax3.axhline(y=200, color="red", linestyle="--", linewidth=2, label="200ms HRI threshold")
ax3.set_title("Full Pipeline Total Latency\n(CPU only — Coral needed)", fontweight="bold")
ax3.set_ylabel("Total Latency (ms)")
ax3.set_ylim(0, 2000)
ax3.legend()
for bar, val in zip(bars3, pipeline_totals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f"{val}ms", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
             f"{val/200:.1f}x\nabove\nthreshold", ha="center", va="center",
             color="white", fontweight="bold", fontsize=9)

plt.tight_layout()
plt.savefig("/catkin_ws/results/cpu_baseline_charts.png", dpi=150, bbox_inches="tight")
print("Chart saved to /catkin_ws/results/cpu_baseline_charts.png")

fig2, ax = plt.subplots(figsize=(10, 6))
categories = ["Face Detection\n(SSD MobileNetV2)", "Emotion Recognition\n(DeepFace)", "Speech-to-Text\n(Whisper tiny.en)", "TOTAL Pipeline"]
cpu_values = [169.0, 1005.2, 546.9, 1721.1]
coral_estimated = [15.0, 1005.2, 546.9, 1567.1]
x = np.arange(len(categories))
width = 0.35
b1 = ax.bar(x - width/2, cpu_values, width, label="CPU only (measured)", color="#1F4E79", edgecolor="white")
b2 = ax.bar(x + width/2, coral_estimated, width, label="With Coral (face estimated)", color="#2E75B6", edgecolor="white", alpha=0.7)
ax.axhline(y=200, color="red", linestyle="--", linewidth=2, label="200ms HRI naturalness threshold")
ax.set_title("CPU vs Coral Estimated Latency\nMiRo Edge AI Dissertation — University of Sheffield 2026",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Latency (ms)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 2000)
plt.tight_layout()
plt.savefig("/catkin_ws/results/cpu_vs_coral_chart.png", dpi=150, bbox_inches="tight")
print("CPU vs Coral chart saved to /catkin_ws/results/cpu_vs_coral_chart.png")
print("All charts generated successfully")
