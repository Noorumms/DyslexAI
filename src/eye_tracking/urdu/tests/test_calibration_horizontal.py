import numpy as np
import pytest

from core import calibration


def session(noise=0.0, seed=0, junk_vertical=False):
    # left-right numbers follow the dot's column; the up-down numbers follow nothing useful
    rng = np.random.default_rng(seed)
    junk = np.random.default_rng(seed + 100)  # its own stream, so the left-right noise is identical
    vectors, targets, dots = [], [], []
    for dot, (x, y) in enumerate(calibration.nine_point_targets(1280, 720)):
        u = -(x / 1280 - 0.5) * 0.16
        for _ in range(30):
            n = rng.normal(0, noise, 4)
            v = junk.normal(0, 0.05, 2) if junk_vertical else [0.0, 0.0]
            vectors.append([u + n[0], v[0], u + n[2], v[1], 0.0, 0.0])
            targets.append((x, y))
            dots.append(dot)
    return vectors, targets, dots


def test_default_models_left_right_only():
    assert calibration.USE_VERTICAL is False


def test_left_right_is_recovered_and_up_down_is_the_middle():
    vectors, targets, dots = session()
    model = calibration.fit_calibration(vectors, targets)
    predicted = calibration.predict_screen(model, vectors)
    assert np.abs(predicted[:, 0] - np.array(targets)[:, 0]).max() < 1.0
    assert np.allclose(predicted[:, 1], 360.0)


def test_junk_up_down_numbers_cannot_hurt_the_left_right_fit():
    clean = calibration.leave_one_dot_out(*session(noise=0.004))
    junk = calibration.leave_one_dot_out(*session(noise=0.004, junk_vertical=True))
    assert junk["mean_x_px"] == pytest.approx(clean["mean_x_px"])


def test_headline_is_the_left_right_error():
    error = calibration.leave_one_dot_out(*session(noise=0.004))
    assert calibration.headline_px(error) == error["mean_x_px"]
    assert error["mean_x_px"] <= error["mean_px"]


def test_headline_is_the_full_error_when_up_down_is_on(monkeypatch):
    monkeypatch.setattr(calibration, "USE_VERTICAL", True)
    error = calibration.leave_one_dot_out(*session(noise=0.004))
    assert calibration.headline_px(error) == error["mean_px"]
