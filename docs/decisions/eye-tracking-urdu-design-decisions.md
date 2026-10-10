# Urdu eye-tracking data collection: design decisions

Written 10 October 2026. Covers the work in `src/eye_tracking/urdu/` up to the word-pointing test.
Later decisions will be added to this note when that work is tested and committed.

DyslexAI is research into reading behaviour. It is not a diagnostic tool.
Groups are TR/SR (typical / struggling reader) or HR/LR (high / low risk).
The word "dyslexic" is used only for a child with a clinical diagnosis.

How to read the evidence labels:

* **CONFIRMED**: we measured it, or read it in a primary source.
* **SECONDARY**: someone else reported it.
* **INFERENCE**: our own reasoning or a simulation. It may be wrong.
* **PROPOSED**: our own choice. Not proven.
* **UNKNOWN**: we do not know yet.

---

## What we started from

We kept the English eye-tracking pipeline written by a teammate as the base, and changed it for
Urdu in small steps. Our own first prototype was retired.

* Her 33 tests all pass. CONFIRMED.
* Our first prototype used a MediaPipe feature that newer versions removed. On a fresh install it
  would crash at the first camera frame. CONFIRMED (checked in our build environment). Her code
  pins the older version that still has it.

## Weaknesses we found in the English pipeline

We read every file, ran her tests, and ran simulations through her real code. The simulations use a
made-up reader whose true stops we know, so these are INFERENCE: the direction is reliable, the
size on a real person is not known.

1. **The camera's frame rate changes the two main features.** The same simulated reader, with sixty
   stops of exactly 250 ms, gave:

   | Frames per second | Stops found (of 60) | Average stop measured |
   |---|---|---|
   | 60 | 60 | 213 ms |
   | 30 | 60 | 200 ms |
   | 24 | 59 | 195 ms |
   | 20 | 55 | 169 ms |
   | 15 | 44 | 229 ms |
   | 10 | 14 | 1089 ms |

   So a dim room that slows the camera could change the features without any change in reading.
2. **A hole in the data looked like a still eye.** Frames with no face were dropped, then speed was
   measured across the hole. In a simulation, 400 ms holes cut 40 stops to 34 and made two stops
   567 ms long (true length 250 ms).
3. **The "usable" bar did not match the page.** It allowed 3 degrees of error, but her lines were only
   1.94 degrees apart. If the vertical error is bell-shaped, about a third of gaze points would land
   on the wrong line at 1 degree of error, and about three quarters at 3 degrees.
4. **The calibration formula had more settings than dot positions** (13 settings per direction from
   8 positions).
5. **Raw files could be overwritten** if a participant name was reused, and a face video was written
   to disk and had to be deleted by hand.
6. **Features included time that was not reading.** The average stop and the reading speed counted
   looking away at the start and end.

Of these, only number 4 is fixed so far (see decision 4 below). The others are not fixed yet.

## Decisions

| # | Decision | Why | Evidence |
|---|---|---|---|
| 1 | Draw the Urdu text in a web browser, and take every word's box from the browser. Do not use text recognition. | OpenCV cannot draw Urdu. A browser shapes Nastaliq correctly and reports exactly where each word landed. Text recognition for Urdu is unreliable. | The page builds on the team laptop at 1280 by 720 (CONFIRMED). Boxes checked by eye in our build environment only. The weakness of text recognition is INFERENCE. |
| 2 | Use **left-right gaze only**. Up-down is a fixed value (the middle of the screen). | The up-down eye number followed the dot rows at 0.81, 0.61 and 0.10 in three recordings, so it cannot be relied on. In the recording where the head stayed still it predicted the row worse than always guessing the middle (262 px against 192 px). | PROPOSED. Three recordings from one person on one laptop. |
| 3 | Show **one line of text per screen**. | With one line on screen we do not need up-down gaze to know which line is being read. | PROPOSED |
| 4 | Fit calibration with a straight line, not the first version's formula with squared terms. | The first formula learned 13 settings per direction from 8 dot positions. A straight line had 15% less error on P001 (331 to 281 px) and 38% less on P002 (281 to 174 px). | PROPOSED. Two recordings. |
| 5 | Measure how close the gaze lands on words with a **word-pointing test**: each word lights up in turn and the person looks at it. | Our only word-accuracy numbers so far come from a simulation. | The test is built. It has not been run on a real person. |

## What we measured

Left-right calibration error on our first four real recordings (one adult, one laptop). Smaller is
better. The target is 3 degrees or less (the team rulebook). The screen width (31 cm) and the
distance to the screen (55 cm) are **estimates, not measurements**, so every number in degrees
could be off by about 10%. CONFIRMED for the numbers, with that caveat.

| Recording | Left-right error |
|---|---|
| P001 | 4.6 degrees |
| P002 | 1.7 degrees |
| P003 | 2.4 degrees |
| P004 | 3.5 degrees |

The best recording (P003) had the laptop raised so the camera was level with the eyes, the chin
resting on a fist, and the light in front of the face.

Estimated share of fixations landing on the right word, from a simulation using our real word
boxes (INFERENCE, not measured on a person):

| Left-right error | Exactly the right word | Right word or its neighbour |
|---|---|---|
| 1.7 degrees | 47% | 86% |
| 2.5 degrees | 32% | 70% |
| 4.6 degrees | 18% | 44% |

## What is not measured or not solved

* How often the gaze lands on the right word on a real person. The word-pointing test is built but
  has never been run.
* Only four recordings, from one person on one laptop.
* The stop-finding settings (30 degrees per second, 80 ms) are the standard ones. They are not
  tuned for a webcam.
* The reading passages were written by an AI assistant. An Urdu teacher has not reviewed them.
* We decided children read aloud, but no audio is recorded yet.
* No child has been recorded. Ethics approval and the consent form are not in place.
* **Biggest open risk:** if struggling readers are recorded at one school and typical readers at
  another, any model learns the school and not the reading. This must be settled with the
  clinician before any recording, because it cannot be fixed afterwards.
