"""I-VT: split a stream of gaze points into fixations and saccades.

A sample is part of a fixation while the eye is moving slower than a velocity
threshold, and part of a saccade once it moves faster. Samples are
(time_ms, x_px, y_px).
"""

import numpy as np

THRESHOLDS = {
    "velocity_deg_s": 30.0,  # standard I-VT cut-off for reading studies
    "min_fixation_ms": 80.0,  # anything shorter is noise, not a real stop
    "merge_gap_ms": 75.0,  # two stops this close together are one fixation
    "merge_dist_px": 30.0,
}


def px_per_degree(screen_w_px, screen_w_cm, distance_cm):
    """One degree of visual angle in pixels, for this desk setup."""
    cm_per_degree = 2 * distance_cm * np.tan(np.radians(0.5))
    return (screen_w_px / screen_w_cm) * cm_per_degree


def velocities(samples, ppd):
    """Point-to-point speed in degrees per second, one value per sample."""
    out = [0.0]
    for (t0, x0, y0), (t1, x1, y1) in zip(samples, samples[1:]):
        dt = (t1 - t0) / 1000.0
        if dt <= 0:
            out.append(out[-1])
            continue
        distance_deg = np.hypot(x1 - x0, y1 - y0) / ppd
        out.append(distance_deg / dt)
    return out


def _runs_below(flags):
    """Start and end indexes of each run of True values."""
    runs = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _summarise(samples, start, end):
    block = samples[start : end + 1]
    xs = [s[1] for s in block]
    ys = [s[2] for s in block]
    return {
        "start_ms": block[0][0],
        "end_ms": block[-1][0],
        "duration_ms": block[-1][0] - block[0][0],
        "x": float(np.mean(xs)),
        "y": float(np.mean(ys)),
        "sample_count": len(block),
    }


def _merge_close(fixations, gap_ms, dist_px):
    merged = []
    for fix in fixations:
        if not merged:
            merged.append(dict(fix))
            continue
        last = merged[-1]
        close_in_time = fix["start_ms"] - last["end_ms"] <= gap_ms
        close_in_space = np.hypot(fix["x"] - last["x"], fix["y"] - last["y"]) <= dist_px
        if close_in_time and close_in_space:
            total = last["sample_count"] + fix["sample_count"]
            last["x"] = (last["x"] * last["sample_count"] + fix["x"] * fix["sample_count"]) / total
            last["y"] = (last["y"] * last["sample_count"] + fix["y"] * fix["sample_count"]) / total
            last["end_ms"] = fix["end_ms"]
            last["duration_ms"] = last["end_ms"] - last["start_ms"]
            last["sample_count"] = total
        else:
            merged.append(dict(fix))
    return merged


def detect_fixations(samples, ppd, thresholds=None):
    if len(samples) < 2:
        return []
    limits = dict(THRESHOLDS)
    if thresholds:
        limits.update(thresholds)

    speeds = velocities(samples, ppd)
    slow = [v < limits["velocity_deg_s"] for v in speeds]
    found = [_summarise(samples, a, b) for a, b in _runs_below(slow)]
    found = _merge_close(found, limits["merge_gap_ms"], limits["merge_dist_px"])
    return [f for f in found if f["duration_ms"] >= limits["min_fixation_ms"]]


def saccades(fixations):
    """The jumps between consecutive fixations."""
    out = []
    for a, b in zip(fixations, fixations[1:]):
        out.append(
            {
                "from_x": a["x"],
                "to_x": b["x"],
                "from_y": a["y"],
                "to_y": b["y"],
                "duration_ms": b["start_ms"] - a["end_ms"],
                "amplitude_px": float(np.hypot(b["x"] - a["x"], b["y"] - a["y"])),
            }
        )
    return out
