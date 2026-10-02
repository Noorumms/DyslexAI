"""Draw the fixations over the passage, numbered in order.

    python scripts/render_scanpath.py P001

Writes data/sessions/P001/scanpath.png. Circle size follows fixation length.
If the path doesn't follow the lines of text, calibration has drifted, even
when the feature numbers look reasonable.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from core import capture

RED = (0, 0, 220)
BLUE = (200, 120, 0)


def draw(image, found):
    previous = None
    for number, fix in enumerate(found, start=1):
        centre = (int(fix["x"]), int(fix["y"]))
        radius = max(6, int(fix["duration_ms"] / 20))
        if previous:
            cv2.line(image, previous, centre, BLUE, 1)
        cv2.circle(image, centre, radius, RED, 2)
        label_at = (centre[0] + radius, centre[1] - radius)
        cv2.putText(image, str(number), label_at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
        previous = centre
    return image


def main(name):
    folder = capture.session_dir(name)
    image = cv2.imread(str(folder / "passage.png"))
    found = json.loads((folder / "fixations.json").read_text())
    out = folder / "scanpath.png"
    cv2.imwrite(str(out), draw(image, found))
    print(f"saved {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/render_scanpath.py P001")
    main(sys.argv[1])
