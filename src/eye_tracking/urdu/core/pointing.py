"""Score a word-pointing test: the person looks at each highlighted word in turn.

We know where they were meant to look, so this measures how close the gaze really lands.
That is the honest answer to "can this camera tell which word?", measured on our own laptop
instead of simulated. Nothing here has been run with a real person yet.
"""

import numpy as np

MIN_SAMPLES = 10  # fewer usable frames than this and the word is skipped, not guessed


def hit_word(x, words):
    """Index of the word box that contains x, or None if x is outside every box."""
    for w in words:
        if w["x0"] <= x < w["x1"]:
            return w["word_index"]
    return None


def score_targets(targets, layout, ppd):
    """targets: [{"word_index": 3, "xs": [predicted x in px, ...]}, ...] for a one-line passage."""
    if len(layout["lines"]) != 1:
        raise ValueError("the pointing test needs a one-line passage")
    words = layout["lines"][0]["words"]
    by_index = {w["word_index"]: w for w in words}

    rows = []
    for t in targets:
        word = by_index[t["word_index"]]
        centre = (word["x0"] + word["x1"]) / 2
        if len(t["xs"]) < MIN_SAMPLES:
            rows.append(
                {
                    "word_index": t["word_index"],
                    "centre_x": centre,
                    "scored": False,
                    "n": len(t["xs"]),
                }
            )
            continue
        median_x = float(np.median(t["xs"]))
        landed = hit_word(median_x, words)
        rows.append(
            {
                "word_index": t["word_index"],
                "centre_x": centre,
                "scored": True,
                "n": len(t["xs"]),
                "median_x": median_x,
                "error_px": median_x - centre,  # negative = the gaze landed left of the word
                "landed_on": landed,
                "exact": landed == t["word_index"],
                "near": landed is not None and abs(landed - t["word_index"]) <= 1,
            }
        )
    return rows, summarise(rows, ppd)


def summarise(rows, ppd):
    scored = [r for r in rows if r["scored"]]
    if len(scored) < 3:
        return {"scored": len(scored), "of": len(rows), "enough": False}
    centres = np.array([r["centre_x"] for r in scored])
    landed = np.array([r["median_x"] for r in scored])
    slope = float(np.polyfit(centres, landed, 1)[0]) if np.ptp(centres) > 0 else float("nan")
    errors = np.array([r["error_px"] for r in scored])
    return {
        "scored": len(scored),
        "of": len(rows),
        "enough": True,
        "exact": sum(r["exact"] for r in scored),
        "near": sum(r["near"] for r in scored),
        "mean_abs_error_px": float(np.abs(errors).mean()),
        "mean_abs_error_deg": float(np.abs(errors).mean() / ppd),
        "mean_shift_px": float(errors.mean()),  # negative = gaze tends to land left of the word
        "range_kept": slope,  # 1.0 = gaze spans the real distance; below 1 = squeezed together
    }
