"""Urdu passage -> page for a browser -> layout.json in the format features.py reads.

The browser draws the text (it shapes Nastaliq correctly, OpenCV cannot draw Urdu at
all) and reports where each word landed. Boxes come from the browser's own layout,
never from OCR. Nothing here has been run against a real child or a real camera.
"""

import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

BUILT = Path(__file__).resolve().parent.parent / "passages" / "built"

# size and spacing are starting values, not tuned: 4 lines about 4.8 deg apart at 55 cm
SETTINGS = {
    "font_px": 90,
    "pitch_px": 210,  # distance between lines; webcam gaze error is large, so lines sit far apart
    "top_px": 60,
    "margin_px": 120,  # text starts this far from the right edge
    "pad_px": 25,  # outer words are widened by this much, like the English space
    "background": "#f0f5f5",
    "ink": "#141414",
}

# these look almost the same as Urdu letters but are Arabic code points: they draw
# nearly identically and quietly break any word list that was typed with the Urdu ones
LOOKALIKES = {
    0x064A: "Arabic yeh",
    0x0643: "Arabic kaf",
    0x0647: "Arabic heh",
    0x0649: "alef maksura",
}

# the browser must be done loading the font before we measure, or boxes are for a fallback font
FONT_READY_JS = "document.fonts.size === 1 && [...document.fonts].every(f => f.status === 'loaded')"
MEASURE_JS = """[...document.querySelectorAll('.w')].map(e => {
  const r = e.getBoundingClientRect();
  return {line: +e.dataset.line, i: +e.dataset.i, text: e.textContent,
          x0: r.left, x1: r.right, y0: r.top, y1: r.bottom};
})"""


def settings_for_screen(width, height, n_lines, base=None):
    """Shrink the whole page in proportion on screens smaller than the one it was designed on.

    The design was checked on 1536x960. A smaller screen gets every size scaled down by the
    same amount, so a line that fitted there still fits. Bigger screens are not scaled up.
    """
    s = {**SETTINGS, **(base or {})}
    scale = min(1.0, width / 1536, height / 960)
    for key in ("font_px", "pitch_px", "top_px", "margin_px", "pad_px"):
        s[key] = round(s[key] * scale)
    room = (height - s["top_px"] - 30) // n_lines  # keep 30 px spare under the last line
    s["pitch_px"] = min(s["pitch_px"], room)
    if n_lines == 1:
        s["top_px"] = (height - s["pitch_px"]) // 2  # a lone line sits in the middle of the screen
    return s


def check_word(word):
    for ch in word:
        if ord(ch) in LOOKALIKES:
            raise ValueError(
                f"{word!r} has {LOOKALIKES[ord(ch)]} (U+{ord(ch):04X}), use the Urdu letter"
            )
        if not (0x0600 <= ord(ch) <= 0x06FF or ord(ch) == 0x200C):
            raise ValueError(f"{word!r} has U+{ord(ch):04X}, which is not in the Arabic block")


def load_passage(path):
    passage = json.loads(Path(path).read_text(encoding="utf-8"))
    if passage.get("language") != "ur":
        raise ValueError("this builder is for Urdu passages (language: ur)")
    for line in passage["lines"]:
        if not line["words"]:
            raise ValueError(f"line {line['line_index']} has no words")
        for word in line["words"]:
            check_word(word)
    return passage


def font_data_url(font_path):
    # the font travels inside the page itself, so the browser can't quietly use another one
    path = Path(font_path)
    kind = {".woff2": ("font/woff2", "woff2"), ".ttf": ("font/ttf", "truetype")}[
        path.suffix.lower()
    ]
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{kind[0]};base64,{data}", kind[1]


def build_html(passage, font_path, width, height, settings=None):
    s = {**SETTINGS, **(settings or {})}
    url, fmt = font_data_url(font_path)
    rows = []
    for i, line in enumerate(passage["lines"]):
        spans = " ".join(
            f'<span class="w" data-line="{i}" data-i="{k}">{html.escape(w)}</span>'
            for k, w in enumerate(line["words"])
        )
        rows.append(
            f'<div class="line" style="top:{s["top_px"] + i * s["pitch_px"]}px">{spans}</div>'
        )
    css = (
        f'@font-face {{ font-family: UrduText; src: url("{url}") format("{fmt}"); }}'
        f"html, body {{ margin: 0; width: {width}px; height: {height}px; overflow: hidden;"
        f" background: {s['background']}; }}"
        f".line {{ position: absolute; right: {s['margin_px']}px; height: {s['pitch_px']}px;"
        f" line-height: {s['pitch_px']}px; white-space: nowrap; font-family: UrduText;"
        f" font-size: {s['font_px']}px; color: {s['ink']}; }}"
    )
    body = "".join(rows)
    return f'<!doctype html><html lang="ur" dir="rtl"><head><meta charset="utf-8"><style>{css}</style></head><body>{body}</body></html>'


def tile_line(spans, pad):
    """Make word boxes meet halfway across each gap, as the English boxes do.

    spans are (x0, x1) in reading order. Nastaliq words can overlap their neighbours, so
    the meeting point is the middle of the overlap (or of the gap), whichever there is.
    """
    order = sorted(range(len(spans)), key=lambda k: spans[k][0] + spans[k][1])
    boxes = [list(s) for s in spans]
    for left, right in zip(order, order[1:]):
        middle = (boxes[left][1] + boxes[right][0]) / 2
        boxes[left][1], boxes[right][0] = middle, middle
    boxes[order[0]][0] -= pad
    boxes[order[-1]][1] += pad
    return boxes


def check_reading_order(spans, line_index):
    # This is the check that makes the RTL claim true: word 0 must be the rightmost
    # word and every later word further left, measured from the real layout.
    centres = [(a + b) / 2 for a, b in spans]
    for k in range(len(centres) - 1):
        if centres[k + 1] >= centres[k]:
            raise ValueError(f"line {line_index}: word {k + 1} is not to the left of word {k}")


def ink_centre(gray, top, bottom):
    """Vertical middle of the dark pixels in a band of rows, measured from the picture."""
    band = gray[int(top) : int(bottom)]
    rows = np.where((band < 128).any(axis=1))[0]
    if len(rows) == 0:
        raise ValueError(f"no text found between rows {top} and {bottom}")
    return top + (np.percentile(rows, 5) + np.percentile(rows, 95)) / 2


def build_layout(passage, measured, gray, width, height, settings=None):
    s = {**SETTINGS, **(settings or {})}
    if s["top_px"] + len(passage["lines"]) * s["pitch_px"] > height:
        raise ValueError(
            "the last line would be cut off: use settings_for_screen() for this screen"
        )
    by_line = {}
    for m in measured:
        by_line.setdefault(m["line"], []).append(m)

    lines, offsets, count = [], [], 0
    for i, line in enumerate(passage["lines"]):
        got = sorted(by_line.get(i, []), key=lambda m: m["i"])
        if [m["text"] for m in got] != line["words"]:
            raise ValueError(f"line {i}: the browser's words differ from the passage file")
        spans = [(m["x0"], m["x1"]) for m in got]
        if min(a for a, _ in spans) < 20 or max(b for _, b in spans) > width - 20:
            raise ValueError(f"line {i} runs off the screen: lower font_px or shorten the line")
        check_reading_order(spans, i)

        top = s["top_px"] + i * s["pitch_px"]
        offsets.append(ink_centre(gray, top, top + s["pitch_px"]) - (top + s["pitch_px"] / 2))
        boxes = tile_line(spans, s["pad_px"])
        words = [
            {"word_index": count + k, "text": m["text"], "x0": float(b[0]), "x1": float(b[1])}
            for k, (m, b) in enumerate(zip(got, boxes))
        ]
        count += len(words)
        lines.append({"line_index": i, "words": words})

    # one shift for the whole page keeps the line bands touching, like the English layout
    shift = float(np.median(offsets))
    one_line = len(lines) == 1
    for line in lines:
        top = s["top_px"] + line["line_index"] * s["pitch_px"]
        if one_line:
            # With one line on screen we know the line without up-down gaze (which does not work
            # on this webcam), so the whole screen height counts as that line.
            line["center_y"], line["half_height"] = height / 2, height / 2
        else:
            line["center_y"] = top + s["pitch_px"] / 2 + shift
            line["half_height"] = s["pitch_px"] / 2

    render = {
        **s,
        "width": width,
        "height": height,
        "ink_shift_px": shift,
        "passage_id": passage["id"],
        "whole_screen_band": one_line,
    }
    return {
        "language": "ur",
        "direction": "rtl",
        "word_count": count,
        "lines": lines,
        "render": render,
    }


def built_stem(passage_id, width, height):
    return BUILT / f"{passage_id}_{width}x{height}"


def load_built(passage_id, width, height):
    """The picture and layout made by scripts/build_urdu_passage.py for exactly this screen."""
    stem = built_stem(passage_id, width, height)
    png, js = stem.with_suffix(".png"), stem.with_suffix(".json")
    if not png.exists() or not js.exists():
        raise FileNotFoundError(
            f"{stem.name} not built for this screen: run scripts/build_urdu_passage.py"
        )
    image = cv2.imread(str(png))
    if image is None or image.shape[:2] != (height, width):
        raise ValueError(f"{png.name} is not {width}x{height}: rebuild it on this laptop")
    return image, json.loads(js.read_text(encoding="utf-8"))
