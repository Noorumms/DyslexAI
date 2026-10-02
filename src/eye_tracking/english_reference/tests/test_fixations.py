from core.fixations import detect_fixations, px_per_degree, velocities

PPD = px_per_degree(1920, 34.5, 60.0)


def stare(start_ms, x, y, count, step_ms=20):
    # a real fixation is never perfectly still, so nudge x by a pixel
    return [(start_ms + i * step_ms, x + (i % 2), y) for i in range(count)]


def test_three_stares_become_three_fixations():
    samples = stare(0, 100, 300, 15)
    samples += stare(300, 400, 300, 15)
    samples += stare(600, 700, 300, 15)
    found = detect_fixations(samples, PPD)
    assert len(found) == 3
    assert all(f["duration_ms"] >= 200 for f in found)
    assert round(found[0]["x"]) in (100, 101)


def test_short_glance_is_dropped():
    samples = stare(0, 100, 300, 15) + stare(300, 500, 300, 2)
    assert len(detect_fixations(samples, PPD)) == 1


def test_fast_sweep_has_no_fixation():
    samples = [(i * 20, i * 300, 300) for i in range(10)]
    assert detect_fixations(samples, PPD) == []


def test_velocity_is_zero_when_still():
    samples = [(0, 100, 100), (20, 100, 100), (40, 100, 100)]
    assert velocities(samples, PPD) == [0.0, 0.0, 0.0]


def test_empty_and_single_sample_give_no_fixations():
    assert detect_fixations([], PPD) == []
    assert detect_fixations([(0, 100, 100)], PPD) == []


def test_duplicate_timestamps_do_not_divide_by_zero():
    samples = [(0, 100, 100), (0, 105, 100), (20, 105, 100)]
    assert len(velocities(samples, PPD)) == 3
