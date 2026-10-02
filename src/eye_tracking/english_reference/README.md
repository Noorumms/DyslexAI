# DyslexAI — M1 eye tracking

Webcam gaze pipeline, offline: MediaPipe landmarks -> calibrated screen point
-> I-VT fixations -> six gaze features and a per-word dwell table.
Read `CLAUDE.md` for how the pipeline fits together and `instructions_FYP.md`
for the project rules.

## Install (Windows, Python 3.12 — not 3.13 or 3.14)

    py -3.12 -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python -m pytest tests/ -q

## Set up this laptop (once per machine)

    copy machine.example.json machine.json
    python scripts/check_setup.py

Sit in front of the camera while it runs. Copy the values it prints into
`machine.json`, then add `screen_w_cm` (visible screen width, ruler) and
`distance_cm` (eye to screen, tape). Every other script refuses to run while
these are zero.

## Record and process one session

Do the first two back to back without moving your chair or head.

    python scripts/capture_calibration.py P001
    python scripts/run_reading_session.py P001
    python scripts/process_session.py P001
    python scripts/render_scanpath.py P001

Everything lands in `data/sessions/P001/`, which git ignores. Open
`scanpath.png` first: if the circles don't follow the lines of text, the
numbers are not worth reading.

For volunteer sessions, delete `reading.mp4` once `process_session.py` has
written `vectors.json` (instructions §8.3). Use P001, P002, never names.
