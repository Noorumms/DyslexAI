import numpy as np

from core.gaze import smooth


def test_smoothing_keeps_a_jump_sharp():
    # a jump between two words must stay a jump, or I-VT sees a slow drift
    # and merges the two fixations
    values = [np.array([0.0])] * 10 + [np.array([1.0])] * 10
    out = [float(v[0]) for v in smooth(values)]
    assert set(out) == {0.0, 1.0}


def test_smoothing_removes_a_single_jitter_spike():
    values = [np.array([0.0])] * 10
    values[5] = np.array([9.0])
    assert all(float(v[0]) == 0.0 for v in smooth(values))
