import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import features, urdu_layout

PASSAGE = Path(__file__).resolve().parent.parent / "passages" / "urdu_easy_1.json"


def fake_measured(passage, width=1536, word_w=100, gap=20):
    # what a browser would report for right-to-left text: word 0 is the rightmost
    out = []
    for i, line in enumerate(passage["lines"]):
        x = width - 120
        for k, word in enumerate(line["words"]):
            out.append(
                {"line": i, "i": k, "text": word, "x0": x - word_w, "x1": x, "y0": 0, "y1": 0}
            )
            x -= word_w + gap
    return out


def fake_page(passage, settings=None, height=960, width=1536):
    s = {**urdu_layout.SETTINGS, **(settings or {})}
    gray = np.full((height, width), 240, dtype=np.uint8)
    for i in range(len(passage["lines"])):
        centre = s["top_px"] + i * s["pitch_px"] + s["pitch_px"] // 2 + 10
        gray[centre - 20 : centre + 20, 200:1400] = 20
    return gray


def test_the_shipped_passage_is_valid():
    passage = urdu_layout.load_passage(PASSAGE)
    assert len(passage["lines"]) == 4
    assert all(1 <= len(line["words"]) <= 10 for line in passage["lines"])


def test_arabic_lookalike_letters_are_refused():
    with pytest.raises(ValueError, match="Arabic kaf"):
        urdu_layout.check_word("\u0643\u062a\u0627\u0628")  # kaf typed with the Arabic code point


def test_latin_letters_are_refused():
    with pytest.raises(ValueError, match="not in the Arabic block"):
        urdu_layout.check_word("abc")


def test_overlapping_nastaliq_words_meet_in_the_middle():
    # three words, the first two overlap by 10 px
    boxes = urdu_layout.tile_line([(900, 1000), (780, 910), (700, 760)], pad=25)
    assert boxes[0][0] == boxes[1][1] == 905  # middle of the overlap
    assert boxes[1][0] == boxes[2][1] == 770  # middle of the gap
    assert boxes[0][1] == 1025 and boxes[2][0] == 675  # outer words widened


def test_left_to_right_layout_is_refused():
    with pytest.raises(ValueError, match="not to the left"):
        urdu_layout.check_reading_order([(100, 200), (300, 400)], line_index=0)


def test_word_zero_is_the_rightmost_and_later_words_go_left():
    passage = urdu_layout.load_passage(PASSAGE)
    layout = urdu_layout.build_layout(
        passage, fake_measured(passage), fake_page(passage), 1536, 960
    )
    first = layout["lines"][0]["words"]
    assert first[0]["word_index"] == 0
    assert first[0]["x0"] > first[1]["x0"] > first[2]["x0"]
    assert layout["word_count"] == 28
    assert layout["lines"][1]["words"][0]["word_index"] == 7  # continues across lines


def test_boxes_leave_no_gap_inside_a_line():
    passage = urdu_layout.load_passage(PASSAGE)
    layout = urdu_layout.build_layout(
        passage, fake_measured(passage), fake_page(passage), 1536, 960
    )
    words = layout["lines"][0]["words"]
    for right, left in zip(words, words[1:]):
        assert left["x1"] == right["x0"]


def test_line_bands_follow_the_measured_ink_and_touch():
    passage = urdu_layout.load_passage(PASSAGE)
    layout = urdu_layout.build_layout(
        passage, fake_measured(passage), fake_page(passage), 1536, 960
    )
    a, b = layout["lines"][0], layout["lines"][1]
    assert a["center_y"] + a["half_height"] == pytest.approx(b["center_y"] - b["half_height"])
    assert layout["render"]["ink_shift_px"] == pytest.approx(
        10, abs=1
    )  # the fake ink sat 10 px low


def test_a_line_that_runs_off_screen_is_refused():
    passage = urdu_layout.load_passage(PASSAGE)
    wide = fake_measured(passage, word_w=200)
    with pytest.raises(ValueError, match="runs off the screen"):
        urdu_layout.build_layout(passage, wide, fake_page(passage), 1536, 960)


def test_browser_words_that_differ_from_the_file_are_refused():
    passage = urdu_layout.load_passage(PASSAGE)
    measured = fake_measured(passage)
    measured[3]["text"] = "غلط"
    with pytest.raises(ValueError, match="differ from the passage file"):
        urdu_layout.build_layout(passage, measured, fake_page(passage), 1536, 960)


def test_blank_page_is_refused():
    with pytest.raises(ValueError, match="no text found"):
        urdu_layout.ink_centre(np.full((300, 300), 240, dtype=np.uint8), 0, 100)


def test_features_read_the_urdu_layout_leftward_is_forward():
    passage = urdu_layout.load_passage(PASSAGE)
    layout = urdu_layout.build_layout(
        passage, fake_measured(passage), fake_page(passage), 1536, 960
    )
    line = layout["lines"][0]
    y = line["center_y"]
    # a reader moving right to left along line 1: the eye's x goes DOWN
    xs = [(w["x0"] + w["x1"]) / 2 for w in line["words"]]
    forward = [
        {"start_ms": 300 * k, "end_ms": 300 * k + 200, "duration_ms": 200, "x": x, "y": y}
        for k, x in enumerate(xs)
    ]
    assert features.count_regressions(features.assign_fixations(forward, layout)) == 0
    # the same fixations visited left to right are all backward steps
    backward = [dict(f, start_ms=300 * k) for k, f in enumerate(reversed(forward))]
    assert features.count_regressions(features.assign_fixations(backward, layout)) == len(xs) - 1


def test_load_built_refuses_a_picture_for_another_screen(tmp_path, monkeypatch):
    monkeypatch.setattr(urdu_layout, "BUILT", tmp_path)
    stem = tmp_path / "urdu_easy_1_1536x960"
    cv2.imwrite(str(stem.with_suffix(".png")), np.zeros((864, 1536, 3), dtype=np.uint8))
    stem.with_suffix(".json").write_text(json.dumps({"lines": []}))
    with pytest.raises(ValueError, match="rebuild it on this laptop"):
        urdu_layout.load_built("urdu_easy_1", 1536, 960)


def test_load_built_says_what_to_run_when_nothing_is_built(tmp_path, monkeypatch):
    monkeypatch.setattr(urdu_layout, "BUILT", tmp_path)
    with pytest.raises(FileNotFoundError, match="build_urdu_passage.py"):
        urdu_layout.load_built("urdu_easy_1", 1536, 960)


def test_a_short_screen_gets_tighter_lines_not_a_cut_off_passage():
    passage = urdu_layout.load_passage(PASSAGE)
    tight = urdu_layout.settings_for_screen(1536, 864, 4)
    assert tight["pitch_px"] < urdu_layout.SETTINGS["pitch_px"]
    measured = fake_measured(passage)
    layout = urdu_layout.build_layout(
        passage, measured, fake_page(passage, tight, height=864), 1536, 864, tight
    )
    last = layout["lines"][-1]
    assert last["center_y"] + last["half_height"] <= 864


def test_a_tall_screen_keeps_the_default_spacing():
    assert (
        urdu_layout.settings_for_screen(1536, 1200, 4)["pitch_px"]
        == urdu_layout.SETTINGS["pitch_px"]
    )


def test_default_spacing_on_a_short_screen_is_refused():
    passage = urdu_layout.load_passage(PASSAGE)
    with pytest.raises(ValueError, match="cut off"):
        urdu_layout.build_layout(
            passage, fake_measured(passage), fake_page(passage, height=864), 1536, 864
        )


def test_a_small_screen_scales_the_font_with_the_page():
    small = urdu_layout.settings_for_screen(1280, 720, 4)
    assert small["font_px"] == 68 and small["pitch_px"] == 158  # 0.75 of the tested design
    assert small["top_px"] + 4 * small["pitch_px"] <= 720


def test_a_bigger_screen_is_not_scaled_up():
    assert (
        urdu_layout.settings_for_screen(1920, 1080, 4)["font_px"] == urdu_layout.SETTINGS["font_px"]
    )


def one_line_passage(i=0):
    full = urdu_layout.load_passage(PASSAGE)
    return {
        "id": f"urdu_line_{i + 1}",
        "language": "ur",
        "lines": [dict(full["lines"][i], line_index=0)],
    }


def test_all_four_one_line_passages_are_valid():
    for i in range(1, 5):
        passage = urdu_layout.load_passage(PASSAGE.parent / f"urdu_line_{i}.json")
        assert len(passage["lines"]) == 1


def test_a_lone_line_sits_in_the_middle_of_a_720_px_screen():
    s = urdu_layout.settings_for_screen(1280, 720, 1)
    assert s["top_px"] + s["pitch_px"] // 2 == pytest.approx(360, abs=1)


def test_a_lone_line_counts_the_whole_screen_height_as_that_line():
    passage = one_line_passage(2)
    s = urdu_layout.settings_for_screen(1280, 720, 1)
    layout = urdu_layout.build_layout(
        passage, fake_measured(passage, 1280), fake_page(passage, s, 720, 1280), 1280, 720, s
    )
    line = layout["lines"][0]
    assert (line["center_y"], line["half_height"]) == (360, 360)
    assert layout["render"]["whole_screen_band"] is True
    # a fixation anywhere up or down still belongs to the one line
    fixation = [{"start_ms": 0, "end_ms": 200, "duration_ms": 200, "x": 900, "y": 20}]
    assert features.assign_fixations(fixation, layout)[0]["line_index"] == 0
