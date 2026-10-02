"""Fixations plus the passage layout -> the six gaze features.

The layout comes from the reading screen, measured after the passage is
rendered, and looks like this:

    {
      "language": "ur",
      "word_count": 11,
      "lines": [
        {"line_index": 0, "center_y": 300.0, "half_height": 45.0,
         "words": [{"word_index": 0, "x0": 900.0, "x1": 990.0}, ...]},
      ]
    }

word_index is the position in READING order across the whole passage, so for
Urdu index 0 is the rightmost word of the first line. Building it that way is
what makes the direction logic below RTL-correct with no language check here.
"""

FEATURE_NAMES = [
    "fixation_count_per_100_words",
    "mean_fixation_ms",
    "regression_count_per_100_words",
    "mean_forward_saccade_words",
    "words_per_minute",
    "line_reread_count",
]


def classify_movement(prev_word_index, next_word_index):
    # In Urdu the reader moves right to left, so the next word in reading
    # order has a HIGHER index but a LOWER x on screen. We compare reading
    # order, not screen x — that is what makes this RTL-correct.
    if next_word_index > prev_word_index:
        return "forward"
    if next_word_index < prev_word_index:
        return "regression"
    return "same"


def find_line(y, layout):
    """The line whose band contains y, or None if the gaze was off the text."""
    for line in layout["lines"]:
        if abs(y - line["center_y"]) <= line["half_height"]:
            return line
    return None


def find_word(x, line):
    for word in line["words"]:
        if word["x0"] <= x <= word["x1"]:
            return word["word_index"]
    return None


def assign_fixations(fixations, layout):
    """Attach a line and word index to each fixation. Both may be None."""
    out = []
    for fix in fixations:
        line = find_line(fix["y"], layout)
        word_index = find_word(fix["x"], line) if line else None
        out.append(
            {
                **fix,
                "line_index": line["line_index"] if line else None,
                "word_index": word_index,
            }
        )
    return out


def word_dwell(assigned):
    """Total time spent on each word — this is the 'stuck on a word' table.

    A word read normally gets one short fixation. A word the reader struggles
    with shows either one long fixation or several returns to it.
    """
    table = {}
    previous = None
    for fix in assigned:
        index = fix["word_index"]
        if index is None:
            previous = None
            continue
        returning = index in table and previous != index
        entry = table.setdefault(
            index, {"dwell_ms": 0.0, "fixation_count": 0, "refixation_count": 0}
        )
        entry["dwell_ms"] += fix["duration_ms"]
        entry["fixation_count"] += 1
        if returning:
            entry["refixation_count"] += 1
        previous = index
    return table


def _word_sequence(assigned):
    return [f["word_index"] for f in assigned if f["word_index"] is not None]


def _line_sequence(assigned):
    return [f["line_index"] for f in assigned if f["line_index"] is not None]


def count_regressions(assigned):
    sequence = _word_sequence(assigned)
    moves = [classify_movement(a, b) for a, b in zip(sequence, sequence[1:])]
    return moves.count("regression")


def count_line_rereads(assigned):
    # a jump back up to a line the reader had already left
    lines = _line_sequence(assigned)
    return sum(1 for a, b in zip(lines, lines[1:]) if b < a)


def mean_forward_saccade(assigned):
    sequence = _word_sequence(assigned)
    steps = [b - a for a, b in zip(sequence, sequence[1:]) if b > a]
    return sum(steps) / len(steps) if steps else 0.0


def reading_time_s(fixations):
    if not fixations:
        return 0.0
    return (fixations[-1]["end_ms"] - fixations[0]["start_ms"]) / 1000.0


def gaze_features(fixations, layout):
    """The six numbers that go into the risk model, one row per session."""
    assigned = assign_fixations(fixations, layout)
    words = max(layout["word_count"], 1)
    on_text = [f for f in assigned if f["word_index"] is not None]
    seconds = reading_time_s(fixations)
    durations = [f["duration_ms"] for f in fixations]

    return {
        "fixation_count_per_100_words": 100.0 * len(on_text) / words,
        "mean_fixation_ms": sum(durations) / len(durations) if durations else 0.0,
        "regression_count_per_100_words": 100.0 * count_regressions(assigned) / words,
        "mean_forward_saccade_words": mean_forward_saccade(assigned),
        # assumes the reader finished the passage, which the reading screen
        # enforces by only advancing on a button press
        "words_per_minute": 60.0 * words / seconds if seconds > 0 else 0.0,
        "line_reread_count": count_line_rereads(assigned),
    }


def quality_score(fixations, frames_seen, frames_with_face, calibration_error_deg):
    """Per §8.5 — sessions below these bars get excluded from training."""
    face_rate = frames_with_face / frames_seen if frames_seen else 0.0
    on_text_time = sum(f["duration_ms"] for f in fixations)
    return {
        "face_detection_rate": face_rate,
        "calibration_error_deg": calibration_error_deg,
        "fixation_time_ms": on_text_time,
        "usable": face_rate >= 0.7 and calibration_error_deg <= 3.0,
    }
