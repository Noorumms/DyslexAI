import numpy as np

from core.calibration import (
    calibration_error,
    fit_calibration,
    leave_one_dot_out,
    nine_point_targets,
)


def fake_session(noise, seed=0):
    # eye numbers generated from a known mapping, 30 frames per dot
    rng = np.random.default_rng(seed)
    vectors, targets, dots = [], [], []
    for dot, (x, y) in enumerate(nine_point_targets(1536, 864)):
        u, v = (x / 1536 - 0.5) * 0.5, (y / 864 - 0.5) * 0.3
        for _ in range(30):
            n = rng.normal(0, noise, 4)
            vectors.append([u + n[0], v + n[1], u + n[2], v + n[3], 0.02, 0.1])
            targets.append((x, y))
            dots.append(dot)
    return vectors, targets, dots


def test_nine_targets_inside_the_screen():
    targets = nine_point_targets(1536, 864)
    assert len(targets) == 9
    assert all(0 < x < 1536 and 0 < y < 864 for x, y in targets)


def test_recovers_a_clean_mapping():
    vectors, targets, dots = fake_session(noise=0.0)
    error = leave_one_dot_out(vectors, targets, dots)
    assert error["mean_px"] < 1.0


def test_held_out_error_is_worse_than_in_sample_error():
    # if leave-one-dot-out leaked, the two numbers would be about the same
    vectors, targets, dots = fake_session(noise=0.03)
    model = fit_calibration(vectors, targets)
    in_sample = calibration_error(model, vectors, targets)["mean_px"]
    held_out = leave_one_dot_out(vectors, targets, dots)["mean_px"]
    assert held_out > in_sample


def test_worst_dot_is_at_least_the_mean():
    vectors, targets, dots = fake_session(noise=0.03)
    error = leave_one_dot_out(vectors, targets, dots)
    assert error["max_px"] >= error["mean_px"]
