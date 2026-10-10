"""Say in plain words why a calibration was good or bad. Reads only, changes nothing.

    python scripts/diagnose_calibration.py P003

The calibration file holds, for each of the nine dots, about 30 frames of six eye numbers.
If your eyes followed the dots, the iris numbers should rise and fall with the dot's position.
The cut-off numbers below are rules of thumb we chose from our own first three recordings,
not published values: treat them as hints, not as proof.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core import calibration, capture, fixations

MIN_FOLLOW = 0.8  # eyes should follow the dot at least this closely
MIN_CLEAR = 3.0  # iris movement across dots compared with jitter inside one dot
MAX_YAW_RANGE = 0.06  # head turn across the nine dots; P002, where the head clearly moved, had 0.17
MAX_PITCH_RANGE = 0.03  # head nod across the nine dots; P002 had 0.04, the still P003 had 0.01


def per_dot_summary(data):
    vectors = np.asarray(data["vectors"], dtype=float)
    targets = np.asarray(data["targets"], dtype=float)
    dots = np.asarray(data["dots"])
    rows = []
    for dot in np.unique(dots):
        mine = dots == dot
        model = calibration.fit_calibration(vectors[~mine], targets[~mine])
        predicted = calibration.predict_screen(model, vectors[mine])
        rows.append(
            {
                "dot": int(dot),
                "target": targets[mine][0],
                "frames": int(mine.sum()),
                "mean": vectors[mine].mean(axis=0),
                "sd": vectors[mine].std(axis=0),
                "error_x": float(np.abs(predicted[:, 0] - targets[mine][:, 0]).mean()),
            }
        )
    return rows


def correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def report(rows, ppd):
    tx = np.array([r["target"][0] for r in rows])
    ty = np.array([r["target"][1] for r in rows])
    means = np.array([r["mean"] for r in rows])
    jitter = np.mean([r["sd"] for r in rows], axis=0)
    iris_x, iris_y = means[:, [0, 2]].mean(axis=1), means[:, [1, 3]].mean(axis=1)
    follow_x, follow_y = correlation(tx, iris_x), correlation(ty, iris_y)
    clear_x = np.ptp(iris_x) / max(jitter[[0, 2]].mean(), 1e-9)
    yaw_range, pitch_range = np.ptp(means[:, 4]), np.ptp(means[:, 5])
    mean_error = np.mean([r["error_x"] for r in rows])

    print("\nLEFT-RIGHT (the part we use)")
    print(f"  followed the dots: {follow_x:.2f}   (want {MIN_FOLLOW} or more)")
    print(f"  clear vs jitter:   {clear_x:.1f}   (want {MIN_CLEAR} or more)")
    print(
        f"  mean error:        {mean_error:.0f} px = {mean_error / ppd:.1f} deg   (want 3 deg or less)"
    )
    print("UP-DOWN (not used: unreliable, it followed the dots in some recordings and not others)")
    print(f"  followed the dots: {follow_y:.2f}   (ours so far: 0.81, 0.61, 0.10)")
    print("HEAD MOVEMENT (how far it moved between dots; smaller is better)")
    print(
        f"  turn: {yaw_range:.3f} (want under {MAX_YAW_RANGE})"
        f"   nod: {pitch_range:.3f} (want under {MAX_PITCH_RANGE})"
    )

    print("\nWhat this suggests:")
    said = False
    if follow_x < MIN_FOLLOW:
        said = True
        print(
            "  - Left-right did not follow the dots. Were you looking at each dot? Re-do it slower."
        )
    if clear_x < MIN_CLEAR:
        said = True
        print(
            "  - The left-right signal is weak next to its jitter. Usually light on the face is too"
        )
        print("    dim or comes from behind you, or the camera is out of focus.")
    if yaw_range > MAX_YAW_RANGE or pitch_range > MAX_PITCH_RANGE:
        said = True
        print(
            "  - Your head moved between dots. Keep it still and move only your eyes. Raising the"
        )
        print(
            "    laptop so the camera is level with your eyes, and resting your chin on your fist, helps."
        )
    worst = max(rows, key=lambda r: r["error_x"])
    if worst["error_x"] > 2 * np.median([r["error_x"] for r in rows]):
        said = True
        x, y = worst["target"]
        print(f"  - One dot is far worse than the rest: dot {worst['dot']} at ({x:.0f}, {y:.0f}).")
        print("    Edge dots are the hardest for a webcam.")
    if not said:
        print("  - Nothing obvious is wrong.")


def main(name):
    machine = capture.load_machine()
    path = capture.session_dir(name) / "calibration.json"
    rows = per_dot_summary(json.loads(path.read_text()))
    ppd = fixations.px_per_degree(
        machine["screen_w_px"], machine["screen_w_cm"], machine["distance_cm"]
    )

    print(f"calibration file: {path}")
    print("\ndot  target(px)   L-R iris  U-D iris    yaw    pitch   L-R error")
    for r in rows:
        m, (x, y) = r["mean"], r["target"]
        print(
            f"{r['dot']:>3}  ({x:>4.0f},{y:>4.0f})   {(m[0] + m[2]) / 2:>+7.3f}   {(m[1] + m[3]) / 2:>+7.3f}"
            f"   {m[4]:>+6.3f}  {m[5]:>+6.3f}   {r['error_x']:>4.0f} px = {r['error_x'] / ppd:.1f} deg"
        )
    report(rows, ppd)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/diagnose_calibration.py P003")
    main(sys.argv[1])
