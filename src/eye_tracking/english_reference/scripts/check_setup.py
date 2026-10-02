"""Stage 0: check this machine before recording any session.

    python scripts/check_setup.py

Prints the Python and MediaPipe versions, the screen size OpenCV really uses,
the webcam's real resolution and frame rate at 480p and 720p, and whether a
face gives sane eye numbers. Sit in front of the camera while it runs.
"""

import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp

from core import capture, gaze


def check_mediapipe():
    print(f"python {platform.python_version()}, mediapipe {mp.__version__}")
    if not hasattr(mp, "solutions"):
        raise SystemExit(
            "this mediapipe has no legacy solutions API: use Python 3.12 and mediapipe==0.10.14"
        )


def measure_camera(index, width, height, seconds=5):
    cap, actual = capture.open_camera(index, width, height)
    times = []
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        if cap.grab():
            times.append((time.perf_counter() - start) * 1000)
    cap.release()
    stats = capture.capture_stats(times)
    print(f"asked {width}x{height}, got {actual[0]}x{actual[1]}")
    print("  " + capture.describe(stats))
    return actual, stats


def check_face(index, width, height):
    cap, _ = capture.open_camera(index, width, height)
    for _ in range(15):
        cap.read()  # first frames are dark while the camera sets its exposure
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("camera gave no frame")

    mesh = gaze.open_face_mesh()
    result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mesh.close()
    if not result.multi_face_landmarks:
        print("no face found: sit in front of the camera in decent light and run again")
        return
    landmarks = result.multi_face_landmarks[0].landmark
    print(f"landmarks: {len(landmarks)} (need 478, which includes iris ids 468 and 473)")
    h, w = frame.shape[:2]
    values = gaze.eye_vector(landmarks, w, h)
    for name, value in zip(gaze.FEATURE_NAMES, values):
        print(f"  {name}: {value:+.3f}")
    print("  iris values should sit roughly between -0.5 and +0.5")


def pick_camera_size(results):
    # 720p only if it really delivers 720p at a usable rate; fewer than 25
    # samples a second is too sparse for velocity-based fixation detection
    size, stats = results[(1280, 720)]
    if size == (1280, 720) and stats.get("fps", 0) >= 25:
        return 1280, 720
    return 640, 480


def main():
    check_mediapipe()
    machine = capture.load_machine()

    w, h = capture.open_fullscreen("screen check")
    cv2.destroyAllWindows()
    print(f"screen as OpenCV sees it: {w}x{h}")

    results = {}
    for size in [(640, 480), (1280, 720)]:
        results[size] = measure_camera(machine["camera_index"], *size)
    cam_w, cam_h = pick_camera_size(results)
    check_face(machine["camera_index"], cam_w, cam_h)

    print("\nput these in machine.json:")
    print(f'  "screen_w_px": {w}, "screen_h_px": {h}, "camera_w": {cam_w}, "camera_h": {cam_h}')
    print('  plus "screen_w_cm" (ruler, visible area) and "distance_cm" (tape, eye to screen)')


if __name__ == "__main__":
    main()
