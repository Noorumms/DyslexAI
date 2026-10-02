"""Show nine dots, record the eye numbers at each, save calibration.json.

    python scripts/capture_calibration.py P001

Keep your head still and follow each dot with your eyes only. Esc aborts.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core import calibration, capture, fixations, gaze

WINDOW = "calibration"
SETTLE_S = 0.6  # the eye needs a moment to land on a new dot
FRAMES_PER_DOT = 30
TIMEOUT_S = 4.0


def dot_image(w, h, x, y):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(image, (int(x), int(y)), 14, (255, 255, 255), -1)
    cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)
    return image


def collect_dot(cap, mesh, image):
    cv2.imshow(WINDOW, image)
    vectors = []
    start = time.perf_counter()
    while time.perf_counter() - start < SETTLE_S + TIMEOUT_S:
        ok, frame = cap.read()
        if cv2.waitKey(1) == 27:
            raise SystemExit("aborted, nothing saved")
        # keep reading during the settle time so old buffered frames are
        # thrown away rather than recorded against the new dot
        if not ok or time.perf_counter() - start < SETTLE_S:
            continue
        vector = gaze.frame_vector(mesh, frame)
        if vector is not None:
            vectors.append(vector.tolist())
        if len(vectors) == FRAMES_PER_DOT:
            break
    return vectors


def run_dots(machine):
    w, h = machine["screen_w_px"], machine["screen_h_px"]
    cap, _ = capture.open_camera(machine["camera_index"], machine["camera_w"], machine["camera_h"])
    mesh = gaze.open_face_mesh()
    data = {"targets": [], "vectors": [], "dots": []}
    try:
        capture.check_screen(WINDOW, machine)
        for dot, (x, y) in enumerate(calibration.nine_point_targets(w, h)):
            got = collect_dot(cap, mesh, dot_image(w, h, x, y))
            if len(got) < FRAMES_PER_DOT // 2:
                raise SystemExit(f"dot {dot}: only {len(got)} frames with a face, check light")
            data["vectors"] += got
            data["targets"] += [[x, y]] * len(got)
            data["dots"] += [dot] * len(got)
    finally:
        cap.release()
        mesh.close()
        cv2.destroyAllWindows()
    return data


def report_error(data, machine):
    ppd = fixations.px_per_degree(
        machine["screen_w_px"], machine["screen_w_cm"], machine["distance_cm"]
    )
    error = calibration.leave_one_dot_out(data["vectors"], data["targets"], data["dots"])
    mean_deg, worst_deg = error["mean_px"] / ppd, error["max_px"] / ppd
    print(
        f"calibration error, leave-one-dot-out: mean {error['mean_px']:.0f} px "
        f"({mean_deg:.2f} deg), worst dot {error['max_px']:.0f} px ({worst_deg:.2f} deg)"
    )
    if mean_deg > 3.0:
        print("above 3 degrees: recalibrate before recording (instructions §8.5)")


def main(name):
    machine = capture.load_machine()
    capture.check_measured(machine)
    data = run_dots(machine)
    out = capture.session_dir(name) / "calibration.json"
    out.write_text(json.dumps(data))
    print(f"saved {out}")
    report_error(data, machine)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/capture_calibration.py P001")
    main(sys.argv[1])
