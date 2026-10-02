import numpy as np
import pytest

from core.capture import (
    capture_stats,
    check_measured,
    pair_with_times,
    read_times,
    write_times,
)


def test_times_round_trip(tmp_path):
    video = tmp_path / "reading.mp4"
    write_times(video, [0.0, 33.3, 66.7])
    assert read_times(video) == [0.0, 33.3, 66.7]


def test_pairing_keeps_real_times_and_drops_frames_without_a_face():
    v = np.zeros(6)
    frames = [(0, v), (1, None), (2, v)]
    paired = pair_with_times(frames, [0.0, 40.0, 90.0])
    assert [t for t, _ in paired] == [0.0, 90.0]


def test_csv_one_row_short_is_refused():
    frames = [(0, None), (1, None), (2, None)]
    with pytest.raises(ValueError):
        pair_with_times(frames, [0.0, 33.0])


def test_capture_stats_sees_a_late_frame():
    stats = capture_stats([0, 33, 66, 116, 149])
    assert stats["interval_max"] == 50
    assert stats["frames"] == 5


def test_unmeasured_machine_is_refused():
    machine = {"screen_w_px": 1536, "screen_h_px": 864, "screen_w_cm": 0, "distance_cm": 60}
    with pytest.raises(ValueError, match="screen_w_cm"):
        check_measured(machine)
