## Live Pipeline Results — Physical MiRo Robot Camera Feed

Test setup: ROS node subscribed to /miro/sensors/caml/compressed (MiRo's native camera topic,
~15 Hz), running SSD MobileNetV2 face detection inline, live against a real person moving in
and out of frame. Two runs, same model, same code path, only the inference backend changed.

| Metric                        | Coral Edge TPU        | CPU-only              |
|--------------------------------|------------------------|------------------------|
| Static single-image inference  | ~8.6 ms / ~116 FPS     | ~54.3 ms / ~18.4 FPS   |
| Live pipeline FPS (sustained)  | 15.0-15.1 FPS (stable) | 15.0-15.8 FPS (unstable) |
| Live pipeline stability        | Zero dips across full run | 2 sustained stalls to ~9.7-10.3 FPS |
| Face detection accuracy        | Correct (verified via annotated snapshot) | Correct (verified via annotated snapshot) |

**Interpretation:** MiRo's camera caps the pipeline at its native ~15 Hz, so raw inference speed
alone does not differentiate the two backends under ideal conditions. The meaningful difference
is headroom and robustness. Coral's ~8.6ms inference leaves ~58ms of slack inside each ~66ms frame
budget, absorbing jitter from decode/ROS overhead with zero observed FPS drops. CPU-only inference
(~54ms) leaves almost no slack, and the live run showed two sustained periods of degraded
throughput (~10 FPS) before recovering. This suggests the Edge TPU's value in this application is
not raw throughput but consistency of real-time performance under load — directly relevant to
social HRI, where inconsistent gaze/attention response would degrade interaction quality even if
average FPS looks acceptable.

Caveat: dia-laptop44 is a shared lab machine; the CPU dips were not cross-checked against
`top`/`htop` output at the time, so a contending process from another team's container cannot be
fully ruled out as a contributing factor. Worth re-running on an idle machine to confirm.

## Gaze-Following Behavior - Live Tracking Precision (Physical MiRo Robot)

Test setup: Direct joint control via /miro/control/kinematic_joints (bypassing MiRo's
higher-level autonomous behaviour layer, after an earlier push-interface approach was found
to trigger an unintended autonomous approach behaviour). Target yaw/pitch computed from the
detected face position using the camera's actual field-of-view constants (CAM_HORI_HALF_FOV,
CAM_VERT_HALF_FOV), clamped to MiRo's physical joint limits, and rate-limited to 5 deg/frame
for smooth, safe motion. Tested with natural continuous head/body movement while seated,
comparing the identical controller driven by Coral-accelerated vs CPU-only face detection.

Fixation timing (Coral, discrete movements): 4 of 5 detected movement episodes converged to
within 5 deg of target within 0.27s-0.65s (mean ~0.48s); the remaining episode was superseded
by a subsequent movement before convergence could be measured, rather than a tracking failure.

Continuous tracking precision:

| Metric                          | Coral Edge TPU  | CPU-only        |
|-----------------------------------|-----------------|-----------------|
| Frames analysed                   | 178             | 167             |
| Mean tracking error                | 0.13 deg        | 0.47 deg        |
| Max (transient) tracking error     | 12.63 deg       | 22.49 deg       |
| Frames within 5 deg of target      | 99%             | 98%             |

**Interpretation:** CPU-only shows roughly 3.6x higher mean tracking error and close to double
the worst-case transient error compared to Coral, consistent with the earlier finding that
CPU-only inference has more irregular frame timing (the FPS stalls measured in the passive
detection comparison). This demonstrates that the acceleration benefit is not confined to raw
inference speed or a synthetic benchmark: it propagates directly into the precision and
stability of the robot's actual physical interactive behaviour, directly supporting the
dissertation's claim of edge acceleration enhancing real-time social human-robot interaction.

Caveat: the two runs were not frame-count-matched (178 vs 167) and natural human movement is
not perfectly repeatable between runs, so this should be read as a representative comparison
under similar conditions rather than a fully controlled, repeated-trials experiment.

## Gaze-Following v2 - Robustness Fix + Distance Variation (Physical MiRo Robot)

Test setup: Same as above (direct joint control, FOV-based targeting, rate-limited to
5 deg/frame), with one addition: an auto-recenter behaviour that returns MiRo to its
calibrated centre position after 30 consecutive frames with no face detected, so the
robot does not remain frozen at a stale off-centre position when the person leaves view.

Trial: continuous single run varying distance from MiRo (approx. 0.5m to 1.5m), solo
(single-face) testing.

| Metric                          | v1 (steady distance) | v2 (distance variation) |
|----------------------------------|-----------------------|--------------------------|
| Total frames                      | -                     | 381                      |
| Frames with face detected         | -                     | 178 (47%)                |
| Frames spent auto-recentering     | -                     | 15 (4%)                  |
| Mean tracking error                | 0.13 deg              | 0.28 deg                 |
| Max (transient) tracking error     | 12.63 deg             | 10.90 deg                |
| Frames within 5 deg of target      | 99%                   | 99%                      |

**Interpretation:** Introducing distance variation lowered the face-detection hit rate
(47%) compared to the steady-distance baseline, since bounding box scale/position shift
more between frames as the subject moves closer/further. This increased mean tracking
error (0.13 -> 0.28 deg), consistent with fewer, more spread-out corrections rather than
smooth continuous ones. However, the auto-recenter behaviour engaged appropriately and
briefly (4% of frames) rather than dominating the run, max error slightly improved, and
99% of detected frames still tracked to within 5 degrees of target - indicating the
controller remains stable and accurate even under a harder, more realistic test
condition, and the robustness fix behaves as intended without masking or degrading core
tracking performance.

Limitation: tested with a single face only (solo testing); multi-face behaviour is
untested and left as a limitation for future work.
