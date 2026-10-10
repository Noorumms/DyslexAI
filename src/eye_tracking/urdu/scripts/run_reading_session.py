"""Show the passage, record the webcam with real frame times.

    python scripts/run_reading_session.py P001            (English test passage)
    python scripts/run_reading_session.py P001 urdu_easy_1   (a passage built for Urdu)

Run straight after capture_calibration.py without moving. Read the passage
once at your normal pace, then press space. Esc aborts.

Writes into data/sessions/P001/: reading.mp4, reading.csv (one real
timestamp per frame), layout.json (word boxes) and passage.png.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from core import calibration, capture, fixations, passage, urdu_layout

WINDOW = "reading"
SPACE, ESC = 32, 27


def record(cap, writer, image):
    cv2.imshow(WINDOW, image)
    times = []
    start = time.perf_counter()
    while True:
        if not cap.grab():
            raise RuntimeError("camera stopped delivering frames")
        # stamp the moment the driver hands the frame over, before decoding
        # and writing, which take a varying amount of time
        stamp = (time.perf_counter() - start) * 1000
        ok, frame = cap.retrieve()
        if ok:
            writer.write(frame)
            times.append(stamp)
        key = cv2.waitKey(1)
        if key == SPACE:
            return times
        if key == ESC:
            raise SystemExit("aborted")


def check_calibration(folder, machine):
    """Stop and ask before recording a reading on top of a calibration that is not good enough."""
    path = folder / "calibration.json"
    if not path.exists():
        raise SystemExit(f"no calibration for {folder.name}: run capture_calibration.py first")
    raw = json.loads(path.read_text())
    error = calibration.leave_one_dot_out(raw["vectors"], raw["targets"], raw["dots"])
    ppd = fixations.px_per_degree(
        machine["screen_w_px"], machine["screen_w_cm"], machine["distance_cm"]
    )
    degrees = calibration.headline_px(error) / ppd
    print(f"calibration error: {degrees:.2f} deg")
    if degrees > 3.0:
        print("That is above 3 degrees, so this recording will be marked not usable.")
        input("Press Enter to record anyway, or Ctrl+C to stop and recalibrate: ")


def main(name, passage_id=None):
    machine = capture.load_machine()
    capture.check_measured(machine)
    folder = capture.session_dir(name)
    check_calibration(folder, machine)
    video = folder / "reading.mp4"
    if video.exists():
        raise SystemExit(f"{video} already exists; use a new session name")

    w, h = machine["screen_w_px"], machine["screen_h_px"]
    if passage_id:
        image, layout = urdu_layout.load_built(passage_id, w, h)
    else:
        image, layout = passage.render_passage(passage.SAMPLE_TEXT, w, h)
    cv2.imwrite(str(folder / "passage.png"), image)
    (folder / "layout.json").write_text(json.dumps(layout, indent=1))

    cap, size = capture.open_camera(
        machine["camera_index"], machine["camera_w"], machine["camera_h"]
    )
    # the fps in the video header is a placeholder; real times go in the csv
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    times = None
    try:
        capture.check_screen(WINDOW, machine)
        times = record(cap, writer, image)
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        if times is None:
            # aborted or crashed: don't leave a half video blocking this name
            video.unlink(missing_ok=True)

    capture.write_times(video, times)
    print(capture.describe(capture.capture_stats(times)))
    print(f"saved to {folder}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: python scripts/run_reading_session.py P001 [urdu_easy_1]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
