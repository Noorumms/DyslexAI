# DyslexAI — Build Instructions

**Project:** DyslexAI — An RTL-Aware Multimodal Webcam-Based Early Dyslexia Screening System for Urdu Readers
**Group:** BSEF23-05 (Noor Fatima, Areeba Saghir, Reham Ali) — PUCIT / FCIT, University of the Punjab
**Supervisor:** Dr. Muhammad Farooq
**This file:** the rulebook for how the project is built. Read it before writing any code, and again before every milestone. If something here conflicts with the approved proposal, the proposal wins and this file gets updated.

---

## 1. Ground rules for code

These apply to every file in the repository. They are not suggestions.

1. **Simple beats clever.** If there are two ways to do something, take the one a third-year student can read out loud in a viva. You will have to defend every line to the FAC.
2. **No complex algorithms.** No custom neural architectures, no attention layers, no deep sequence models, no hand-rolled optimisers, no heavy design patterns. Velocity-threshold (I-VT) fixation detection, a polynomial fit for gaze mapping, and an SVM or logistic regression for scoring are all this project needs.
3. **Write code that looks like a person wrote it.** Short functions, ordinary names (`gaze_points`, `fix_list`, `words`), normal spacing, occasional short comment where the reason isn't obvious. No 20-line docstrings, no emojis, no decorative banner comments, no `# ============` separators, no "Enhanced/Advanced/Ultimate" in names.
4. **No unnecessary code.** Do not write a class when a function does. Do not add config options nobody sets. Do not write abstract base classes, factories, dependency-injection containers, plugin registries, or a generic "framework" for a 3-person FYP. Delete anything unused the same day you notice it.
5. **No dead code, no commented-out blocks.** Git has the history.
6. **Every file under ~300 lines, every function under ~40 lines.** If it grows past that, split it along an obvious seam, not an invented one.
7. **Comments explain *why*, not *what*.** `# iris center is noisy, average the last 5 frames` is useful. `# loop through frames` is not.
8. **Consistency over personal taste.** One style across the three of you: `snake_case` in Python, `camelCase` in JS, 4-space indent in Python, 2 in JS. Run `black` on Python before every commit and be done with the argument.
9. **If you did not write it, you must still be able to explain it.** Anything pasted from a tutorial, Stack Overflow, or an AI assistant gets read line by line, renamed to match our style, and stripped of whatever it does that we don't need. Code you cannot explain in the demo is a liability, not a feature.

### Style example

Not this:

```python
class AbstractGazeProcessorFactory:
    """
    Enterprise-grade factory for instantiating gaze processing strategies
    with configurable backends and pluggable smoothing pipelines.
    ...
    """
    def create(self, backend: str = "mediapipe", **kwargs) -> "BaseGazeProcessor":
        ...
```

This:

```python
def smooth(vectors, window=5):
    # a median, not an average: an average smears jumps between words and
    # I-VT then merges neighbouring fixations
    out = []
    for i in range(len(vectors)):
        chunk = vectors[max(0, i - window + 1) : i + 1]
        out.append(np.median(chunk, axis=0))
    return out
```

---

## 2. What we are building (and what we are not)

### In scope

| Module | Owner | What it does |
|---|---|---|
| M1 Eye tracking | Noor | Webcam → MediaPipe face/iris landmarks → calibrated gaze point → fixations → RTL-aware gaze features |
| M2 Speech | Reham | Read-aloud recording → Whisper transcript → word error rate, words per minute, pause count |
| M3 Behavioural quiz | Areeba | Short Urdu/English self- or parent-report questionnaire → score |
| M4 Risk scoring | Noor + Reham | Combine M1–M3 features → one classifier → risk band + feature explanation |
| M5 Dashboard | Areeba | React screens: session setup, calibration, reading, quiz, report |
| Data | All three | Phase 1 adult PUCIT volunteer sessions + ETDD70 for validating the English pipeline |

### Out of scope — do not start these

- Handwriting / OCR analysis. The proposal explicitly excludes it. (Note: earlier project notes described a handwriting-image approach; the approved proposal replaced it with gaze + speech. Follow the proposal.)
- Gamified learning interface (M5 in the proposal) — stretch goal only, and only after D3 is fully working.
- Data collection from children. Phase 1 is adult volunteers, which is what keeps us out of clinical-ethics overhead. Any child data needs written ethics approval first; if the supervisor pushes for it, that is a separate conversation, not a sprint task.
- Mobile app, multi-tenant accounts, payment, cloud deployment, Docker Swarm/Kubernetes, CI pipelines beyond a single GitHub Action running tests.
- Any claim, in UI or documentation, that the system diagnoses dyslexia. It produces a **screening risk indicator**. This wording is fixed everywhere: UI, report PDF, slides, paper.

---

## 3. Stack — the simple version

The proposal lists several options per layer. Pick one and stop deliberating:

| Layer | Use | Why |
|---|---|---|
| CV / gaze | Python 3.12, opencv-contrib-python, MediaPipe 0.10.14 legacy `solutions` API (`refine_landmarks=True`), NumPy | Already in the proposal; iris landmarks come free. 0.10.14 has no build for Python 3.13+ |
| Speech | `faster-whisper`, `small` model, `language="ur"` / `"en"` | `small` runs on a laptop CPU; `large` does not |
| ML | scikit-learn only (SVM, LogisticRegression, StandardScaler, GroupKFold) + `shap` | No PyTorch unless something forces it, and nothing will |
| Backend | **FastAPI only** | One backend, one language, one set of bugs |
| ASP.NET Core | Only if the department requires it for the "enterprise backend" marks. If so, keep it to login + session CRUD, and let FastAPI own everything AI. Ask the supervisor before dropping it. |
| Database | SQLite via SQLAlchemy for development; PostgreSQL only if deployment needs it | Zero setup, one file, easy to back up and demo. Confirm with the supervisor since the proposal says MongoDB/Firebase/SQL Server. |
| Frontend | React + Vite, plain CSS modules or Tailwind | No Redux, no Next.js, no component library beyond what you actually use |
| Tools | Git + GitHub, Jira, Postman | As proposed |

Deviations from the proposal's stack (single backend, SQLite) must be raised with Dr. Farooq once, in writing, with the reason. Do not silently diverge — the proposal is a marked document.

---

## 4. Repository layout

```text
dyslexai/
├── README.md
├── instructions.md              # this file
├── requirements.txt
├── machine.example.json         # copy to machine.json per laptop (gitignored)
├── backend/
│   ├── main.py                  # FastAPI app, routes only
│   ├── db.py                    # SQLAlchemy engine + models
│   ├── schemas.py               # pydantic request/response models
│   └── routes/
│       ├── sessions.py
│       ├── gaze.py
│       ├── speech.py
│       └── score.py
├── core/                        # all the actual logic, no web code in here
│   ├── calibration.py
│   ├── capture.py               # camera, real frame timestamps, machine.json
│   ├── gaze.py                  # landmarks -> gaze point
│   ├── fixations.py             # gaze points -> fixations (I-VT)
│   ├── passage.py               # English test passage + word-box layout
│   ├── features.py              # fixations + layout -> feature dict
│   ├── speech.py                # audio -> wer, wpm, pauses
│   ├── quiz.py
│   └── model.py                 # train / predict / explain
├── scripts/                     # thin command-line wrappers around core/
│   ├── check_setup.py
│   ├── capture_calibration.py
│   ├── run_reading_session.py
│   ├── process_session.py
│   └── render_scanpath.py
├── passages/
│   ├── urdu_easy_1.json
│   ├── urdu_medium_1.json
│   └── ...                      # text, fixed line breaks, word list
├── data/
│   ├── sessions/                # raw per-session csv/json (gitignored)
│   └── features.csv             # one row per session, committed only if anonymous
├── notebooks/
│   └── etdd70_check.ipynb       # exploration only, never imported by app code
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── pages/               # Setup, Calibrate, Read, Quiz, Report
│       └── components/
├── tests/
└── docs/
    ├── srs.md
    ├── architecture.md
    └── consent_form.md
```

Rule: `core/` never imports from `backend/`. Logic is testable without a server running. That single rule saves the integration sprint.

---

## 5. Build order

Build in this order. Do not start a phase before the previous one runs end to end. "Runs" means a person who is not you can follow the README and get it working.

### Phase 0 — Setup (Week 1)
- Repo, branch rules (`main` protected, work on `feature/...`, PR reviewed by one teammate), `.gitignore` covering `data/sessions/`, `venv`, `node_modules`, `*.wav`.
- `requirements.txt` pinned. Everyone on the same Python version.
- One-page `README.md`: how to install, how to run backend, how to run frontend.
- Download ETDD70 from Zenodo, open it, write `docs/etdd70_notes.md` describing what columns exist. **Do this before writing a single line of gaze code** — it tells you what features are realistic.

### Phase 1 — Gaze pipeline offline (Weeks 2–5) → D1
Work with recorded video and saved CSVs first. Do not build a UI yet.
1. `core/gaze.py`: MediaPipe face mesh with iris refinement → for each frame, produce iris centre positions normalised by eye-corner distance, plus head yaw/pitch. ~80 lines.
2. `core/calibration.py`: show 9 points, collect samples, fit `x = f(features)` and `y = g(features)` with `sklearn.LinearRegression` on the six eye numbers and their squares. Report error leave-one-dot-out (fit on 8 dots, predict the 9th, every dot), never with frames from a test dot in the fit. ~60 lines.
3. `core/fixations.py`: velocity-threshold detection (I-VT), as the proposal specifies. Samples moving slower than the threshold (30 deg/s) are fixation, faster are saccade; merge close fixations, drop ones under 80 ms. Timestamps come from the per-frame CSV, never the video header. Smooth with a median, not an average (see §1 example). Thresholds go in a small dict at the top of the file so they can be tuned, not buried.
4. `core/features.py`: map fixations to lines (nearest line centre in y), then to words (which word box contains x, given the line). Compute the feature list in §6.
5. Validate on ETDD70: does the pipeline produce sane numbers on known data?

**D1 deliverable:** calibration + fixation detection working on video, a notebook showing gaze plotted over a passage image, ETDD70 notes, repo set up.

### Phase 2 — Speech and quiz (Weeks 6–9) → D2
1. `core/speech.py`: record → `faster-whisper` transcript with word timestamps → compare with the expected passage using `difflib.SequenceMatcher` → count substitutions/deletions/insertions → WER, WPM, pause count (gaps > 1 s between word timestamps). No custom alignment algorithm; `difflib` is in the standard library and is enough.
2. `core/quiz.py`: 15–20 Likert items in a JSON file, sum to a score, three bands. That is the whole module — do not over-build it.
3. Urdu passages authored and stored as JSON with fixed line breaks (see §7).
4. Start Phase 1 data collection: 10 adult volunteer sessions.

**D2 deliverable:** speech module and quiz working, 10 sessions collected, per-session feature CSV being written.

### Phase 3 — Integration and scoring (Weeks 10–14) → D3
1. FastAPI endpoints wrapping `core/` — thin routes, no logic in them.
2. React screens: setup → calibrate → read passage → quiz → report.
3. `core/model.py`: `StandardScaler` + `SVC(probability=True)` in a `Pipeline`, trained with `GroupKFold` grouped by participant. SHAP `KernelExplainer` on the pipeline for feature attribution; if SHAP is too slow on the laptop, fall back to logistic-regression coefficients and say so in the report.
4. Reach 30 sessions.

**D3 deliverable:** full pipeline integrated, 30 Urdu sessions, cross-validated numbers reported honestly.

### Phase 4 — Polish, evaluation, documentation (Weeks 15–18) → D4
Deployment, demo video, poster, final documentation, and the paper draft if time allows. No new features after Week 15. None.

---

## 6. Module rules

### M1 — Eye tracking (RTL)

The RTL inversion is the project's contribution, so it must be **one clearly named, well-commented function** that a examiner can point at:

```python
def classify_movement(prev_word_index, next_word_index):
    # In Urdu the reader moves right to left, so the next word in reading
    # order has a HIGHER index but a LOWER x on screen. We compare reading
    # order, not screen x — that is what makes this RTL-correct.
    if next_word_index > prev_word_index:
        return "forward"
    if next_word_index < prev_word_index:
        return "regression"
    return "same"
```

Store word indexes in **reading order** (0 = rightmost word on the line for Urdu, leftmost for English) when the passage JSON is built. Then every direction rule above works for both languages with no `if language == "urdu"` scattered through the code. One flag at passage-build time, none at analysis time.

Keep the feature list short — 8 to 10 features, all explainable:

`fixation_count_per_100_words`, `mean_fixation_ms`, `median_fixation_ms`, `regression_count_per_100_words`, `mean_forward_saccade_words`, `reading_time_s`, `words_per_minute`, `line_reread_count`, `off_text_ratio`.

Do not add a feature you cannot explain in one sentence to Dr. Farooq.

**Be honest about accuracy.** Webcam gaze is accurate to a couple of degrees at best. That means line-level and timing features are reliable; word-level features are approximate; character-level is impossible. Build word boxes generously (large font, wide line spacing, max 8–10 words per line) and say in the report that word assignment is approximate.

### M2 — Speech

- Save audio as 16 kHz mono WAV. Start recording before the passage appears; log the timestamp so gaze and audio share a clock.
- Whisper's Urdu output will contain mistakes. Do not treat it as ground truth. For the pilot sessions, one team member listens and corrects the transcript; store both `asr_text` and `checked_text` and use the checked one for training.
- Rule-based fluency banding is fine and is what the proposal promises. Do not train a second model for this.

### M3 — Quiz

Plain JSON questions, plain sum, plain bands. Bilingual strings in the same file. That's it.

### M4 — Risk scoring

- Train on **session-level** rows: one row per session, ~12–15 columns.
- **Split by participant, never by row.** This is a hard rule carried over from earlier project work and it applies identically here: several sessions or passages from the same person in both train and test will produce a fake 95% and it will fall apart in the demo. Use `GroupKFold(groups=participant_id)`.
- With 30 sessions you cannot report a single train/test accuracy. Report cross-validated mean ± standard deviation, and report the number of participants next to it. Every time.
- The proposal's 85% and 92% targets are aspirations taken from lab-grade studies. If our real number is 70%, we write 70% and explain why. A defended honest result passes; an unexplainable inflated one does not.
- Output is a **risk band** (low / moderate / elevated) plus the top 3 contributing features in plain language, not a bare percentage.

### M5 — Dashboard

Five screens, no more: Setup, Calibration, Reading, Quiz, Report. Urdu rendering needs `dir="rtl"`, `lang="ur"`, Noto Nastaliq Urdu font, font size ≥ 36 px, line height ~2.4, no text justification, fixed line breaks from the passage JSON. Never let the browser reflow the text — the word boxes recorded at render time must match what the reader saw.

---

## 7. Passage files

One JSON per passage. Build the word boxes from the rendered DOM (`getBoundingClientRect()` on each word span) and save them with the session, because box positions depend on screen size.

```json
{
  "id": "urdu_easy_1",
  "language": "ur",
  "level": "easy",
  "lines": [
    {"line_index": 0, "words": ["علی", "ہر", "روز", "اسکول", "جاتا", "ہے"]},
    {"line_index": 1, "words": ["اس", "کا", "بستہ", "نیلا", "ہے"]}
  ],
  "questions": [
    {"q": "علی کہاں جاتا ہے؟", "options": ["اسکول", "بازار", "باغ"], "answer": 0}
  ]
}
```

Word index inside a line is reading order: index 0 is the **rightmost** word for Urdu. Use Urdu code points, not Arabic look-alikes (ی U+06CC, ک U+06A9, ہ U+06C1, ے U+06D2, ۔ U+06D4). A wrong code point renders differently and breaks the word list.

---

## 8. Data rules

1. Participant-level splits everywhere. See M4.
2. Never store a volunteer's name, roll number, CNIC or phone in the dataset. Use `P001`, `P002`. Keep the consent forms in a separate folder that is not in the repo.
3. Raw webcam video is not saved. Save landmarks and gaze points only. Say this on the consent form, then honour it.
4. Written consent before every session, even for adult volunteers. A one-page Urdu/English form in `docs/consent_form.md`.
5. Log a quality score per session (face detection rate, calibration error, valid-gaze ratio). Sessions with calibration error above ~3° or valid gaze below 70% get marked and excluded from training. Keep them in the raw folder with the reason.
6. Never show a volunteer their score. We are not qualified to deliver one and the model is not validated.

---

## 9. Working rhythm

- Two-week sprints as proposed. Each sprint ends with something that runs, not with a slide.
- Daily 10-minute standup; weekly written summary to the supervisor.
- One PR per feature, reviewed by one teammate. The reviewer's job is to ask "can I explain this in the viva?" — if not, it goes back.
- Jira backlog seeded from §5. Tasks are sized in hours, not story points.
- Every module gets 2–4 small tests in `tests/` covering the parts most likely to break silently: fixation detection on a hand-made sample, word-index direction logic, WER on a known transcript pair. Not a coverage target — just the traps.

**Definition of done for any task:** code merged, runs from a clean checkout, has a test or a documented manual check, and the owner can explain it without opening the file.

---

## 10. Things that will sink this project

Read this list at the start of every sprint.

1. Building the UI before the gaze pipeline works offline.
2. Training a model on 8 participants and celebrating 97%.
3. Splitting data by row instead of by participant.
4. Adding the gamified module before D3 is done.
5. Whisper on Urdu being treated as accurate without anyone listening to a recording.
6. Silently changing the stack from the approved proposal.
7. Chasing the 92% number in the proposal instead of reporting what we actually got.
8. Writing clever code nobody on the team can defend in the final evaluation.
9. Collecting data before the consent form and the passage files are final — those sessions get thrown away.
10. Leaving documentation to the last week.

---

## Change log

- 2026-09-22: Fixation detection changed from I-DT to I-VT to match the approved proposal. Python 3.12 and the pinned MediaPipe version recorded. Calibration error is now leave-one-dot-out. Smoothing changed from average to median after a simulation showed the average merged most fixations. `scripts/` and `machine.example.json` added to the layout.

*Keep this file updated. If the supervisor changes direction, edit here first, then change the code.*
