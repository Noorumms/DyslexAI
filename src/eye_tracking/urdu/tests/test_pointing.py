import numpy as np
import pytest

from core import pointing

PPD = 40.0


def layout(n=7, width=100, gap=0):
    words, x = [], 1200.0
    for k in range(n):  # right to left, word 0 on the right
        words.append({"word_index": k, "text": "w", "x0": x - width, "x1": x})
        x -= width + gap
    return {"lines": [{"line_index": 0, "center_y": 360, "half_height": 360, "words": words}]}


def targets(lay, shift=0.0, squeeze=1.0, noise=0.0, n=30, seed=0):
    rng = np.random.default_rng(seed)
    words = lay["lines"][0]["words"]
    mid = np.mean([(w["x0"] + w["x1"]) / 2 for w in words])
    out = []
    for w in words:
        c = (w["x0"] + w["x1"]) / 2
        out.append(
            {
                "word_index": w["word_index"],
                "xs": list(mid + (c - mid) * squeeze + shift + rng.normal(0, noise, n)),
            }
        )
    return out


def test_perfect_gaze_hits_every_word():
    lay = layout()
    rows, s = pointing.score_targets(targets(lay), lay, PPD)
    assert s["exact"] == 7 and s["near"] == 7
    assert s["mean_abs_error_px"] == pytest.approx(0, abs=1e-6)
    assert s["range_kept"] == pytest.approx(1.0)


def test_a_constant_shift_is_reported_with_its_sign():
    lay = layout()
    _, s = pointing.score_targets(targets(lay, shift=-30), lay, PPD)
    assert s["mean_shift_px"] == pytest.approx(-30)
    assert s["mean_abs_error_px"] == pytest.approx(30)


def test_squeezed_range_is_visible_as_range_kept_below_one():
    lay = layout()
    _, s = pointing.score_targets(targets(lay, squeeze=0.6), lay, PPD)
    assert s["range_kept"] == pytest.approx(0.6)
    assert s["exact"] < 7  # the outer words are missed when the range shrinks


def test_one_word_off_counts_as_near_but_not_exact():
    lay = layout()
    _, s = pointing.score_targets(targets(lay, shift=100), lay, PPD)  # exactly one word width
    assert s["exact"] == 0 and s["near"] == 7 - 1  # the end word's neighbour is off the text


def test_gaze_outside_the_text_counts_as_neither():
    lay = layout()
    rows, s = pointing.score_targets(targets(lay, shift=-2000), lay, PPD)
    assert s["exact"] == 0 and s["near"] == 0
    assert all(r["landed_on"] is None for r in rows)


def test_a_word_with_too_few_frames_is_skipped_not_guessed():
    lay = layout()
    t = targets(lay)
    t[2]["xs"] = t[2]["xs"][:3]
    rows, s = pointing.score_targets(t, lay, PPD)
    assert rows[2]["scored"] is False
    assert s["scored"] == 6 and s["of"] == 7


def test_too_few_scored_words_gives_no_summary():
    lay = layout()
    t = [dict(x, xs=x["xs"][:2]) for x in targets(lay)]
    _, s = pointing.score_targets(t, lay, PPD)
    assert s["enough"] is False


def test_a_multi_line_passage_is_refused():
    lay = layout()
    lay["lines"].append(dict(lay["lines"][0], line_index=1))
    with pytest.raises(ValueError, match="one-line"):
        pointing.score_targets(targets(layout()), lay, PPD)


def test_noise_lowers_the_exact_hit_rate():
    lay = layout()
    _, quiet = pointing.score_targets(targets(lay, noise=10), lay, PPD)
    _, loud = pointing.score_targets(targets(lay, noise=120, n=200), lay, PPD)
    assert quiet["exact"] >= loud["exact"]
