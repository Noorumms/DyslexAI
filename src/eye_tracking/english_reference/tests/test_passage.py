import pytest

from core.passage import MAX_WORDS_PER_LINE, SAMPLE_TEXT, render_passage


def all_words(layout):
    return [w for line in layout["lines"] for w in line["words"]]


def test_every_word_is_in_the_layout_in_reading_order():
    _, layout = render_passage(SAMPLE_TEXT, 1536, 864)
    words = all_words(layout)
    assert [w["text"] for w in words] == SAMPLE_TEXT.split()
    assert [w["word_index"] for w in words] == list(range(len(words)))
    assert layout["word_count"] == len(words)


def test_no_line_has_more_than_the_limit():
    _, layout = render_passage(SAMPLE_TEXT, 1536, 864)
    assert all(len(line["words"]) <= MAX_WORDS_PER_LINE for line in layout["lines"])


def test_boxes_meet_with_no_gap_and_english_runs_left_to_right():
    _, layout = render_passage(SAMPLE_TEXT, 1536, 864)
    for line in layout["lines"]:
        for left, right in zip(line["words"], line["words"][1:]):
            assert left["x1"] == right["x0"]
            assert left["x0"] < right["x0"]


def test_line_bands_do_not_overlap():
    _, layout = render_passage(SAMPLE_TEXT, 1536, 864)
    lines = layout["lines"]
    for upper, lower in zip(lines, lines[1:]):
        assert (
            upper["center_y"] + upper["half_height"]
            <= lower["center_y"] - lower["half_height"] + 1e-6
        )


def test_everything_fits_on_screen():
    image, layout = render_passage(SAMPLE_TEXT, 1536, 864)
    assert image.shape == (864, 1536, 3)
    for w in all_words(layout):
        assert 0 <= w["x0"] and w["x1"] <= 1536


def test_passage_too_long_for_screen_is_refused():
    with pytest.raises(ValueError):
        render_passage(SAMPLE_TEXT * 5, 1536, 864)
