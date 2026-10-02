"""Draw an English passage with OpenCV and record where every word landed.

This is a test harness for the gaze pipeline. Urdu Nastaliq needs proper
shaping and will be rendered by the React screen, not here.
"""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 1.6  # about 36 px tall, the minimum in instructions §6
THICKNESS = 2
LINE_SPACING = 2.4
MAX_WORDS_PER_LINE = 8
MARGIN = 0.08
BACKGROUND = (240, 245, 245)
INK = (20, 20, 20)

SAMPLE_TEXT = (
    "Every morning Sara walks to the small market near her home. She buys "
    "fresh bread, a few apples and a bottle of milk. The shopkeeper knows her "
    "name and always asks about her family. On the way back she stops at the "
    "park to watch the birds by the pond. Then she goes home and makes tea "
    "for her grandmother."
)


def text_width(word):
    return cv2.getTextSize(word, FONT, FONT_SCALE, THICKNESS)[0][0]


def wrap_words(words, max_width):
    space = text_width(" ")
    lines, current, width = [], [], 0
    for word in words:
        extra = text_width(word) + (space if current else 0)
        if current and (width + extra > max_width or len(current) == MAX_WORDS_PER_LINE):
            lines.append(current)
            current, width = [], 0
            extra = text_width(word)
        current.append(word)
        width += extra
    if current:
        lines.append(current)
    return lines


def line_boxes(words, x_start, first_index):
    """Word boxes for one line, widened to meet halfway across each gap.

    Webcam gaze is off by a word's width or more, so a fixation that lands in
    the space between two words should still count for the nearer one.
    """
    space = text_width(" ")
    edges, x = [], x_start
    for word in words:
        edges.append([x, x + text_width(word)])
        x += text_width(word) + space
    for left, right in zip(edges, edges[1:]):
        middle = (left[1] + right[0]) / 2
        left[1], right[0] = middle, middle
    edges[0][0] -= space
    edges[-1][1] += space
    return [
        {"word_index": first_index + i, "text": w, "x0": float(a), "x1": float(b)}
        for i, (w, (a, b)) in enumerate(zip(words, edges))
    ]


def draw_line(image, words, x_start, baseline):
    # draw word by word at the same x the boxes use; drawing the whole line in
    # one call drifts up to ~30 px from the summed word widths by the line end
    space = text_width(" ")
    x = x_start
    for word in words:
        cv2.putText(image, word, (int(x), baseline), FONT, FONT_SCALE, INK, THICKNESS)
        x += text_width(word) + space


def render_passage(text, screen_w, screen_h):
    """Return the image to show and the layout that features.py reads."""
    (_, text_h), _ = cv2.getTextSize("Hg", FONT, FONT_SCALE, THICKNESS)
    pitch = text_h * LINE_SPACING
    x_start = int(screen_w * MARGIN)
    lines = wrap_words(text.split(), screen_w - 2 * x_start)
    if pitch * (len(lines) + 1) > screen_h:
        raise ValueError(f"{len(lines)} lines will not fit on a {screen_h} px tall screen")

    image = np.full((screen_h, screen_w, 3), BACKGROUND, dtype=np.uint8)
    layout = {"language": "en", "word_count": 0, "lines": []}
    for line_index, words in enumerate(lines):
        baseline = int(pitch * (line_index + 1))
        boxes = line_boxes(words, x_start, layout["word_count"])
        draw_line(image, words, x_start, baseline)
        layout["lines"].append(
            {
                "line_index": line_index,
                "center_y": baseline - text_h / 2,
                "half_height": pitch / 2,
                "words": boxes,
            }
        )
        layout["word_count"] += len(words)
    return image, layout
