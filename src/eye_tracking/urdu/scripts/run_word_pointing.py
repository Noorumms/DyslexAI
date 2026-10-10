"""Look at each highlighted word in turn, to measure how close the gaze really lands.

    python scripts/capture_calibration.py P005
    python scripts/run_word_pointing.py P005 urdu_line_1

Use the SAME name for both, one straight after the other, without moving. Each word of the
one-line passage gets a yellow block for about two seconds: look at the middle of the block
and do not read ahead. SPACE starts, Esc aborts.

Saves no video. Writes data/sessions/P005/pointing.json with the raw eye numbers per word and
the scores. Nothing here has been run with a real person yet.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core import calibration, capture, fixations, gaze, pointing, urdu_layout

WINDOW = "pointing"
SPACE, ESC = 32, 27
HOLD_S = 2.0  # each word stays highlighted this long
SETTLE_S = 0.7  # the eye needs a moment to land; frames in this time are thrown away


def highlighted(image, word, y_centre, half_height):
    shade = image.copy()
    top, bottom = int(y_centre - half_height), int(y_centre + half_height)
    cv2.rectangle(shade, (int(word["x0"]), top), (int(word["x1"]), bottom), (0, 200, 255), -1)
    return cv2.addWeighted(shade, 0.35, image, 0.65, 0)


def wait_for_space(image):
    shown = image.copy()
    cv2.putText(
        shown, "Look at the yellow block. SPACE to start", (40, 60), 0, 1.0, (60, 60, 60), 2
    )
    cv2.imshow(WINDOW, shown)
    while True:
        key = cv2.waitKey(30)
        if key == SPACE:
            return
        if key == ESC:
            raise SystemExit("aborted")


def look_at(cap, mesh, image):
    """Show one highlighted word; return (times in ms, eye numbers) for the settled part."""
    cv2.imshow(WINDOW, image)
    start = time.perf_counter()
    times, vectors = [], []
    while time.perf_counter() - start < HOLD_S:
        ok, frame = cap.read()
        elapsed = time.perf_counter() - start
        if cv2.waitKey(1) == ESC:
            raise SystemExit("aborted")
        if not ok or elapsed < SETTLE_S:
            continue
        vector = gaze.frame_vector(mesh, frame)
        if vector is not None:
            times.append(elapsed * 1000)
            vectors.append(vector.tolist())
    return times, vectors


def print_report(rows, summary):
    print("\nword  middle(px)  gaze landed(px)  off by     landed on word")
    for r in rows:
        if not r["scored"]:
            print(f"{r['word_index']:>4}  {r['centre_x']:>8.0f}   (only {r['n']} frames: skipped)")
            continue
        landed = "outside the text" if r["landed_on"] is None else f"word {r['landed_on']}"
        print(
            f"{r['word_index']:>4}  {r['centre_x']:>8.0f}  {r['median_x']:>13.0f}  "
            f"{r['error_px']:>+7.0f} px   {landed}"
        )
    if not summary["enough"]:
        print(f"\nonly {summary['scored']} of {summary['of']} words scored: too few to summarise")
        return
    s = summary
    print(f"\nexactly the right word: {s['exact']} of {s['scored']}")
    print(f"right word or a neighbour: {s['near']} of {s['scored']}")
    print(
        f"average distance from the word's middle: {s['mean_abs_error_px']:.0f} px = {s['mean_abs_error_deg']:.1f} deg"
    )
    print(f"average shift: {s['mean_shift_px']:+.0f} px (negative = gaze lands left of the word)")
    print(
        f"range kept: {s['range_kept']:.0%} (100% = gaze covers the real distance between the words)"
    )


def main(name, passage_id):
    machine = capture.load_machine()
    capture.check_measured(machine)
    folder = capture.session_dir(name)
    out = folder / "pointing.json"
    if out.exists():
        raise SystemExit(f"{out} already exists; use a new session name")
    cal_path = folder / "calibration.json"
    if not cal_path.exists():
        raise SystemExit(f"no calibration for {name}: run capture_calibration.py {name} first")

    raw = json.loads(cal_path.read_text())
    model = calibration.fit_calibration(raw["vectors"], raw["targets"])
    ppd = fixations.px_per_degree(
        machine["screen_w_px"], machine["screen_w_cm"], machine["distance_cm"]
    )
    cal_error = calibration.leave_one_dot_out(raw["vectors"], raw["targets"], raw["dots"])
    print(
        f"calibration (left-right, leave-one-dot-out): {calibration.headline_px(cal_error) / ppd:.2f} deg"
    )

    image, layout = urdu_layout.load_built(
        passage_id, machine["screen_w_px"], machine["screen_h_px"]
    )
    line = layout["lines"][0]
    render = layout["render"]
    y_centre = render["top_px"] + render["pitch_px"] / 2 + render["ink_shift_px"]

    cap, _ = capture.open_camera(machine["camera_index"], machine["camera_w"], machine["camera_h"])
    mesh = gaze.open_face_mesh()
    recorded = []
    try:
        capture.check_screen(WINDOW, machine)
        wait_for_space(image)
        for word in line["words"]:  # reading order: word 0 is the rightmost
            shown = highlighted(image, word, y_centre, render["pitch_px"] / 2)
            times, vectors = look_at(cap, mesh, shown)
            xs = calibration.predict_screen(model, vectors)[:, 0].tolist() if vectors else []
            recorded.append(
                {"word_index": word["word_index"], "times_ms": times, "vectors": vectors, "xs": xs}
            )
    finally:
        cap.release()
        mesh.close()
        cv2.destroyAllWindows()

    rows, summary = pointing.score_targets(
        [{"word_index": r["word_index"], "xs": r["xs"]} for r in recorded], layout, ppd
    )
    print_report(rows, summary)
    out.write_text(
        json.dumps({"passage": passage_id, "rows": rows, "summary": summary, "raw": recorded})
    )
    print(f"\nsaved {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python scripts/run_word_pointing.py P005 urdu_line_1")
    main(sys.argv[1], sys.argv[2])
