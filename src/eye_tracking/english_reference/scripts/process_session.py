"""Turn one recorded session into its feature row.

    python scripts/process_session.py P001

Reads data/sessions/P001/ and writes vectors.json (eye numbers per frame, so
the video can be deleted) and fixations.json (for render_scanpath.py).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core import calibration, capture, features, fixations, gaze


def calibrate(folder, ppd):
    raw = json.loads((folder / "calibration.json").read_text())
    error = calibration.leave_one_dot_out(raw["vectors"], raw["targets"], raw["dots"])
    # the error is estimated with dots held out; the model we actually use
    # is then fitted on all nine
    model = calibration.fit_calibration(raw["vectors"], raw["targets"])
    return model, error["mean_px"] / ppd


def load_samples(folder):
    cache = folder / "vectors.json"
    if cache.exists():
        return json.loads(cache.read_text())
    video = folder / "reading.mp4"
    times = capture.read_times(video)
    paired = capture.pair_with_times(gaze.video_vectors(str(video)), times)
    data = {
        "frames_seen": len(times),
        "capture": capture.capture_stats(times),
        "samples": [[t, v.tolist()] for t, v in paired],
    }
    cache.write_text(json.dumps(data))
    print(f"eye numbers saved to {cache}")
    print("for a volunteer session, delete reading.mp4 now (instructions §8.3)")
    return data


def to_screen(samples, model):
    if not samples:
        return []
    times = [s[0] for s in samples]
    vectors = [np.array(s[1]) for s in samples]
    points = calibration.predict_screen(model, gaze.smooth(vectors))
    return [(t, float(p[0]), float(p[1])) for t, p in zip(times, points)]


def print_report(data, error_deg, found, layout, quality):
    print(capture.describe(data["capture"]))
    print(f"face detection rate: {quality['face_detection_rate']:.1%}")
    print(f"calibration error (leave-one-dot-out): {error_deg:.2f} deg")
    print(f"usable for training: {quality['usable']}")
    print(
        f"gaze samples: {len(data['samples'])}  fixations: {len(found)}  "
        f"words: {layout['word_count']}"
    )
    for name, value in features.gaze_features(found, layout).items():
        print(f"  {name}: {value:.2f}")

    words = {w["word_index"]: w["text"] for line in layout["lines"] for w in line["words"]}
    dwell = features.word_dwell(features.assign_fixations(found, layout))
    print("slowest words:")
    for index, row in sorted(dwell.items(), key=lambda kv: -kv[1]["dwell_ms"])[:5]:
        print(
            f"  {words.get(index, index)}: {row['dwell_ms']:.0f} ms, "
            f"{row['fixation_count']} fixations, {row['refixation_count']} returns"
        )


def main(name):
    machine = capture.load_machine()
    capture.check_measured(machine)
    folder = capture.session_dir(name)
    ppd = fixations.px_per_degree(
        machine["screen_w_px"], machine["screen_w_cm"], machine["distance_cm"]
    )
    model, error_deg = calibrate(folder, ppd)
    data = load_samples(folder)
    found = fixations.detect_fixations(to_screen(data["samples"], model), ppd)
    (folder / "fixations.json").write_text(json.dumps(found, indent=1))

    layout = json.loads((folder / "layout.json").read_text())
    quality = features.quality_score(found, data["frames_seen"], len(data["samples"]), error_deg)
    print_report(data, error_deg, found, layout, quality)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/process_session.py P001")
    main(sys.argv[1])
