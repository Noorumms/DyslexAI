"""Draw an Urdu passage in a browser, save the picture and the box around every word.

    python scripts/build_urdu_passage.py urdu_easy_1

Run it once on each laptop: the boxes only match when the picture is shown at the same
pixel size as that screen. One-time setup, needs internet once:

    pip install playwright
    playwright install chromium

Writes passages/built/urdu_easy_1_<w>x<h>.png and .json, and a _check.png. Open the
_check.png and look: every box must sit on its word and every band on its line.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

from core import capture, fixations, urdu_layout

REPO = Path(__file__).resolve().parent.parent
FONT_DIR = REPO / "assets" / "fonts"


def find_font():
    found = [p for p in sorted(FONT_DIR.glob("*")) if "nastaliq" in p.name.lower()]
    found = [p for p in found if p.suffix.lower() in (".woff2", ".ttf")]
    if not found:
        raise FileNotFoundError(f"put a Noto Nastaliq Urdu .woff2 or .ttf in {FONT_DIR}")
    return found[0]


def render(page_html, width, height):
    """Return the screenshot bytes, every word's measured box, and the browser version."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.set_content(page_html)
        # without this the browser may measure a fallback font and nothing would warn us
        page.wait_for_function(urdu_layout.FONT_READY_JS, timeout=10000)
        measured = page.evaluate(urdu_layout.MEASURE_JS)
        png = page.screenshot()
        version = browser.version
        browser.close()
    return png, measured, version


def draw_check(image, layout):
    out = image.copy()
    for line in layout["lines"]:
        top = int(line["center_y"] - line["half_height"])
        bottom = int(line["center_y"] + line["half_height"])
        cv2.rectangle(out, (0, top), (out.shape[1] - 1, bottom), (200, 160, 0), 1)
        for w in line["words"]:
            cv2.rectangle(out, (int(w["x0"]), top + 6), (int(w["x1"]), bottom - 6), (0, 140, 0), 1)
            cv2.putText(
                out, str(w["word_index"]), (int(w["x1"]) - 26, top + 24), 0, 0.5, (0, 0, 200), 1
            )
    return out


def main(passage_id):
    machine = capture.load_machine()
    width, height = machine["screen_w_px"], machine["screen_h_px"]
    if not (width and height):
        raise SystemExit(
            "fill in screen_w_px and screen_h_px in machine.json (check_setup.py prints them)"
        )

    passage = urdu_layout.load_passage(REPO / "passages" / f"{passage_id}.json")
    settings = urdu_layout.settings_for_screen(width, height, len(passage["lines"]))
    page_html = urdu_layout.build_html(passage, find_font(), width, height, settings)

    png, measured, version = render(page_html, width, height)
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if image.shape[:2] != (height, width):
        raise SystemExit(
            f"screenshot is {image.shape[1]}x{image.shape[0]}, expected {width}x{height}"
        )
    layout = urdu_layout.build_layout(
        passage, measured, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), width, height, settings
    )
    layout["render"]["browser"] = version  # kept so a later change in boxes can be explained

    stem = urdu_layout.built_stem(passage_id, width, height)
    stem.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(stem.with_suffix(".png")), image)
    stem.with_suffix(".json").write_text(
        json.dumps(layout, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    cv2.imwrite(str(stem) + "_check.png", draw_check(image, layout))

    print(f"built {stem.name}: {layout['word_count']} words, browser {version}")
    if len(passage["lines"]) > 1 and machine.get("screen_w_cm") and machine.get("distance_cm"):
        ppd = fixations.px_per_degree(width, machine["screen_w_cm"], machine["distance_cm"])
        print(f"line spacing {settings['pitch_px']} px = {settings['pitch_px'] / ppd:.1f} deg")
    print(f"now open {stem.name}_check.png and look at it")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/build_urdu_passage.py urdu_easy_1")
    main(sys.argv[1])
