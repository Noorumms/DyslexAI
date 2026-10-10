"""Turn the eye numbers into a screen coordinate."""

import numpy as np
from sklearn.linear_model import LinearRegression

# What the fit may use. Decided on our first three real calibrations (one person, one laptop,
# so thin evidence: PROPOSED, to be re-checked with held-out validation dots).
#
# - The first version used all six numbers plus their squares: 13 settings per axis learned from
#   only 8 different dot positions. A straight line on the iris numbers had 20-35% less
#   leave-one-dot-out error on sessions P001 and P002.
# - Up-down was unreliable between recordings: the up-down iris number followed the dot rows at
#   0.81 (P001), 0.61 (P002) and 0.10 (P003, the session where the head stayed still). In P003 it
#   followed the dot's column instead, and predicted the row worse than always guessing the middle
#   (262 px against 192 px). Webcam up-down gaze is known to be hard because the eyelids move with
#   the eye (INFERENCE). So up-down is not modelled: y is a constant.
# - Left-right did work (it followed the dots at 0.98 in P003).
# Design consequence: show ONE line of text per screen, so up-down is not needed to know the line.
USE_VERTICAL = False
HORIZONTAL_COLUMNS = [0, 2]  # left_x, right_x
BOTH_AXES_COLUMNS = [0, 1, 2, 3]  # the four iris numbers; only used if USE_VERTICAL


def nine_point_targets(screen_w, screen_h, margin=0.1):
    """The nine dots shown during calibration, in reading order."""
    xs = [margin, 0.5, 1 - margin]
    ys = [margin, 0.5, 1 - margin]
    return [(x * screen_w, y * screen_h) for y in ys for x in xs]


def fit_calibration(vectors, targets):
    """Fit left-right from the iris numbers; up-down is the middle of the dots unless switched on."""
    v = np.asarray(vectors, dtype=float)
    t = np.asarray(targets, dtype=float)
    if USE_VERTICAL:
        cols = BOTH_AXES_COLUMNS
        return {
            "columns": cols,
            "x": LinearRegression().fit(v[:, cols], t[:, 0]),
            "y": LinearRegression().fit(v[:, cols], t[:, 1]),
        }
    cols = HORIZONTAL_COLUMNS
    return {
        "columns": cols,
        "x": LinearRegression().fit(v[:, cols], t[:, 0]),
        "y_constant": float(t[:, 1].mean()),
    }


def predict_screen(model, vectors):
    v = np.asarray(vectors, dtype=float)[:, model["columns"]]
    x = model["x"].predict(v)
    y = model["y"].predict(v) if "y" in model else np.full(len(v), model["y_constant"])
    return np.column_stack([x, y])


def calibration_error(model, vectors, targets):
    """Mean and worst distance in pixels on held-out validation dots."""
    predicted = predict_screen(model, vectors)
    distances = np.linalg.norm(predicted - np.asarray(targets, dtype=float), axis=1)
    return {"mean_px": float(distances.mean()), "max_px": float(distances.max())}


def error_in_degrees(error_px, px_per_degree):
    return error_px / px_per_degree


def leave_one_dot_out(vectors, targets, dots):
    """Honest calibration error: fit on eight dots, predict the ninth, repeat.

    Frames from one dot are near-identical, so if any of them are in the fit
    the model has effectively already seen the answer. Hold out whole dots.
    Returns the mean error across dots and the error at the worst dot, both for the
    full distance and for the left-right part alone.
    """
    vectors = np.asarray(vectors, dtype=float)
    targets = np.asarray(targets, dtype=float)
    dots = np.asarray(dots)
    per_dot, per_dot_x = [], []
    for dot in np.unique(dots):
        test = dots == dot
        model = fit_calibration(vectors[~test], targets[~test])
        predicted = predict_screen(model, vectors[test])
        per_dot.append(np.linalg.norm(predicted - targets[test], axis=1).mean())
        per_dot_x.append(np.abs(predicted[:, 0] - targets[test][:, 0]).mean())
    return {
        "mean_px": float(np.mean(per_dot)),
        "max_px": float(np.max(per_dot)),
        "mean_x_px": float(np.mean(per_dot_x)),
        "max_x_px": float(np.max(per_dot_x)),
    }


def headline_px(error):
    """The error that decides whether a calibration is good enough: left-right if up-down is off."""
    return error["mean_px"] if USE_VERTICAL else error["mean_x_px"]
