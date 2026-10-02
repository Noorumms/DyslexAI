"""Camera, real frame timestamps, and the per-machine desk settings."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
MEASURED_KEYS = ["screen_w_px", "screen_h_px", "screen_w_cm", "distance_cm"]


def load_machine():
    path = REPO / "machine.json"
    if not path.exists():
        # one per laptop, never committed: every machine measures its own
        raise FileNotFoundError("copy machine.example.json to machine.json first")
    return json.loads(path.read_text())


def check_measured(machine):
    # these four drive px_per_degree, which sets the fixation threshold, so a
    # guessed value silently shifts every number in the report
    missing = [key for key in MEASURED_KEYS if not machine.get(key)]
    if missing:
        raise ValueError(f"fill in machine.json first, still zero: {', '.join(missing)}")


def open_camera(index, width, height):
    """Open the webcam and ask for a size. Returns the camera and the size it
    actually gave, because webcams silently ignore sizes they can't do."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return cap, actual


def session_dir(name):
    folder = REPO / "data" / "sessions" / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def open_fullscreen(window):
    """Open a fullscreen OpenCV window and return the size it really got.

    On Windows with display scaling this is the scaled-down logical size
    (1536x864 on a 1920x1080 screen at 125%), and that is the coordinate
    space every gaze point and word box in this project lives in.
    """
    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window, np.zeros((10, 10, 3), dtype=np.uint8))
    cv2.waitKey(300)
    _, _, w, h = cv2.getWindowImageRect(window)
    return w, h


def check_screen(window, machine):
    size = open_fullscreen(window)
    expected = (machine["screen_w_px"], machine["screen_h_px"])
    if size != expected:
        cv2.destroyAllWindows()
        raise ValueError(
            f"fullscreen window is {size[0]}x{size[1]} but machine.json says "
            f"{expected[0]}x{expected[1]}; run scripts/check_setup.py again"
        )


def times_path(video_path):
    # reading.mp4 always pairs with reading.csv, so sessions can't get mixed up
    return Path(video_path).with_suffix(".csv")


def write_times(video_path, times_ms):
    with open(times_path(video_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "time_ms"])
        for i, t in enumerate(times_ms):
            writer.writerow([i, f"{t:.3f}"])


def read_times(video_path):
    with open(times_path(video_path), newline="") as f:
        return [float(row["time_ms"]) for row in csv.DictReader(f)]


def pair_with_times(indexed_vectors, times_ms):
    """Match (frame_index, vector) pairs to real timestamps, keeping faces only.

    The video header's fps is not trusted anywhere; webcams deliver frames
    unevenly. A count mismatch means the CSV belongs to a different recording.
    """
    indexed_vectors = list(indexed_vectors)
    if len(indexed_vectors) != len(times_ms):
        raise ValueError(
            f"video has {len(indexed_vectors)} frames but the CSV has {len(times_ms)} rows"
        )
    return [(times_ms[i], v) for i, v in indexed_vectors if v is not None]


def capture_stats(times_ms):
    if len(times_ms) < 2:
        return {"frames": len(times_ms), "fps": 0.0, "interval_ms": 0.0, "interval_sd": 0.0}
    gaps = np.diff(times_ms)
    return {
        "frames": len(times_ms),
        "seconds": (times_ms[-1] - times_ms[0]) / 1000.0,
        "fps": 1000.0 / gaps.mean(),
        "interval_ms": float(gaps.mean()),
        "interval_sd": float(gaps.std()),
        "interval_max": float(gaps.max()),
    }


def describe(stats):
    if stats["frames"] < 2:
        return "capture: too few frames"
    return (
        f"capture: {stats['frames']} frames over {stats['seconds']:.1f} s, "
        f"{stats['fps']:.1f} fps, interval {stats['interval_ms']:.1f} ms "
        f"(sd {stats['interval_sd']:.1f}, max {stats['interval_max']:.1f})"
    )
