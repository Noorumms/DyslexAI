from core.features import (
    assign_fixations,
    classify_movement,
    count_line_rereads,
    count_regressions,
    gaze_features,
    word_dwell,
)

# Two Urdu lines. Word index 0 is the RIGHTMOST word, so screen x DECREASES
# as the reading index increases. Any logic that reads direction off x alone
# will get these backwards, which is the whole point of the test.
LAYOUT = {
    "language": "ur",
    "word_count": 6,
    "lines": [
        {
            "line_index": 0,
            "center_y": 200.0,
            "half_height": 40.0,
            "words": [
                {"word_index": 0, "x0": 800, "x1": 900},
                {"word_index": 1, "x0": 650, "x1": 750},
                {"word_index": 2, "x0": 500, "x1": 600},
            ],
        },
        {
            "line_index": 1,
            "center_y": 320.0,
            "half_height": 40.0,
            "words": [
                {"word_index": 3, "x0": 800, "x1": 900},
                {"word_index": 4, "x0": 650, "x1": 750},
                {"word_index": 5, "x0": 500, "x1": 600},
            ],
        },
    ],
}


def fix(start, duration, x, y):
    return {
        "start_ms": start,
        "end_ms": start + duration,
        "duration_ms": duration,
        "x": x,
        "y": y,
        "sample_count": 5,
    }


def test_reading_order_decides_direction_not_screen_x():
    assert classify_movement(0, 1) == "forward"
    assert classify_movement(2, 1) == "regression"
    assert classify_movement(1, 1) == "same"


def test_leftward_move_in_urdu_is_forward_not_regression():
    # 850 px -> 700 px is a move LEFT on screen, which would be a regression
    # in English and is normal forward reading in Urdu
    found = [fix(0, 200, 850, 200), fix(250, 200, 700, 200)]
    assert count_regressions(assign_fixations(found, LAYOUT)) == 0


def test_rightward_move_in_urdu_is_a_regression():
    found = [fix(0, 200, 550, 200), fix(250, 200, 850, 200)]
    assert count_regressions(assign_fixations(found, LAYOUT)) == 1


def test_jump_back_to_previous_line_counts_as_reread():
    found = [fix(0, 200, 850, 320), fix(250, 200, 700, 200)]
    assert count_line_rereads(assign_fixations(found, LAYOUT)) == 1


def test_word_dwell_separates_long_stare_from_returns():
    found = [
        fix(0, 200, 850, 200),
        fix(250, 900, 700, 200),
        fix(1200, 200, 550, 200),
        fix(1450, 300, 700, 200),
    ]
    dwell = word_dwell(assign_fixations(found, LAYOUT))
    assert dwell[1]["dwell_ms"] == 1200
    assert dwell[1]["fixation_count"] == 2
    assert dwell[1]["refixation_count"] == 1


def test_off_text_fixation_is_ignored():
    found = [fix(0, 200, 850, 200), fix(250, 200, 100, 900)]
    assigned = assign_fixations(found, LAYOUT)
    assert assigned[1]["line_index"] is None
    assert assigned[1]["word_index"] is None


def test_feature_row_has_the_six_features():
    found = [fix(i * 250, 200, x, 200) for i, x in enumerate([850, 700, 550])]
    row = gaze_features(found, LAYOUT)
    assert len(row) == 6
    assert row["mean_fixation_ms"] == 200
    assert row["fixation_count_per_100_words"] == 50.0
    assert row["mean_forward_saccade_words"] == 1.0


def test_fixation_in_gap_between_boxes_is_not_given_to_nearest_word():
    # 775 px is between word 1 (650-750) and word 0 (800-900)
    assigned = assign_fixations([fix(0, 200, 775, 200)], LAYOUT)
    assert assigned[0]["word_index"] is None


def test_leaving_the_text_and_coming_back_counts_as_a_return():
    found = [fix(0, 200, 700, 200), fix(250, 200, 100, 900), fix(500, 200, 700, 200)]
    dwell = word_dwell(assign_fixations(found, LAYOUT))
    assert dwell[1]["fixation_count"] == 2
    assert dwell[1]["refixation_count"] == 1


def test_no_fixations_gives_zeros_not_a_crash():
    row = gaze_features([], LAYOUT)
    assert all(value == 0 for value in row.values())
