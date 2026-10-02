"""Turn the six eye numbers into a screen coordinate."""

import numpy as np
from sklearn.linear_model import LinearRegression


def nine_point_targets(screen_w, screen_h, margin=0.1):
    """The nine dots shown during calibration, in reading order."""
    xs = [margin, 0.5, 1 - margin]
    ys = [margin, 0.5, 1 - margin]
    return [(x * screen_w, y * screen_h) for y in ys for x in xs]


def design_matrix(vectors):
    # squared terms let the fit bend slightly near the screen edges, where a
    # straight-line mapping is visibly off
    v = np.asarray(vectors, dtype=float)
    return np.hstack([v, v**2])


def fit_calibration(vectors, targets):
    """Fit one model for x and one for y. Needs a few samples per dot."""
    x_design = design_matrix(vectors)
    targets = np.asarray(targets, dtype=float)
    return {
        "x": LinearRegression().fit(x_design, targets[:, 0]),
        "y": LinearRegression().fit(x_design, targets[:, 1]),
    }


def predict_screen(model, vectors):
    design = design_matrix(vectors)
    return np.column_stack([model["x"].predict(design), model["y"].predict(design)])


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
    Returns the mean error across dots and the error at the worst dot.
    """
    vectors = np.asarray(vectors, dtype=float)
    targets = np.asarray(targets, dtype=float)
    dots = np.asarray(dots)
    per_dot = []
    for dot in np.unique(dots):
        test = dots == dot
        model = fit_calibration(vectors[~test], targets[~test])
        predicted = predict_screen(model, vectors[test])
        per_dot.append(np.linalg.norm(predicted - targets[test], axis=1).mean())
    return {"mean_px": float(np.mean(per_dot)), "max_px": float(np.max(per_dot))}
