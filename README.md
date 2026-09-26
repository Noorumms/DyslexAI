# DyslexAI

> A research project investigating computational analysis of reading and writing
> difficulty in Urdu-speaking children.

**DyslexAI does not diagnose dyslexia.** It is a research-oriented screening and
risk-indication tool. Clinical diagnosis remains with qualified professionals.

BS Software Engineering Final Year Design Project
Punjab University College of Information Technology (PUCIT), University of the Punjab, Lahore

---

## Status

🚧 **Research and early implementation.** No integrated application exists yet.

The project is planned as multimodal. The modalities are at very different stages,
and this table reflects what has actually been built rather than what is planned.

| Modality | Status | Detail |
|---|---|---|
| **Handwriting** | Active implementation | Public datasets audited; cleaning, splitting and preprocessing pipeline built; baseline CNN trained and evaluated |
| **Eye tracking / gaze** | Research complete, implementation blocked | Reference datasets reconstructed from primary sources; webcam feasibility established; prototype built but **not runtime-tested**. Blocked on participant access |
| **Speech / oral reading** | Feasibility in progress | Phase 0 and Stage 1 complete. Arm not finalised |
| **Behavioural screening** | Not started | Named as planned scope; no work performed |
| **Application / interface** | Not started | Requirements not yet defined |

System requirements are still being defined. Until they are, this repository
contains **research and pipeline engineering**, not an application.

---

## What is actually in this repository

Notebooks, audit artefacts, research logs and figures for the handwriting modality,
plus the retrospective research record for the eye-tracking and speech investigations.

There is no runnable screening system in this repository yet.

---

## Repository structure

```
DyslexAI/
├── notebooks/
│   └── handwriting/         Research and experiment notebooks, executed with outputs
├── datasets/
│   └── handwriting/
│       ├── metadata/        Master metadata: one row per image, with SHA-256
│       ├── manifests/       Dataset versions and the removal log
│       └── splits/          Train/validation/test manifests
├── docs/
│   ├── reports/             Generated audit and cleaning reports
│   ├── research/            Retrospective research logs (Aug–Sep 2026)
│   └── decisions/           Recorded scope and methodology decisions
├── results/
│   └── figures/             Figures referenced in reports and the dissertation
├── src/                     Source code (application, not yet populated)
├── app/                     Application (not yet populated)
├── tests/                   Tests (not yet populated)
└── assets/                  Static assets
```

**No image data is committed.** See *Data and ethics* below.

---

## Handwriting modality — current findings

The public dataset used for the pilot was audited before any model was trained on it.
These are measured values, reproducible from the committed notebooks and metadata.

### Dataset audit

| Measure | Value |
|---|---|
| Image files supplied | 852 |
| Unique SHA-256 hashes | 638 |
| Redundant copies | 214 |
| Files with contradictory labels | 48 across 20 groups |
| Usable images after cleaning | 618 |
| True class balance after cleaning | 406 / 212 (≈1.92 : 1) |

The dataset's advertised 426/426 class balance is produced by **duplicating files in one
class**, not by collecting 852 distinct samples. A number of images appear under both
class labels simultaneously.

### Measured split leakage

| Split | Test images duplicated in training |
|---|---|
| Supplied data, random split | **36.72%** |
| Cleaned data, random split | 0% |
| Cleaned, capture session held out | 0% |

### Baseline experiment

Identical model, hyperparameters and random seed across three data conditions.
Only the data differs.

| Condition | Balanced accuracy | Majority baseline |
|---|---|---|
| Cleaned, random split | 0.635 | 0.656 |
| Raw supplied, random split | 0.602 | 0.500 |
| Cleaned, **session held out** | **0.515** | 0.541 |

**Principal finding.** Performance falls to chance when the test set comes from a capture
session the model has never seen. This suggests that most of what the model learns on a
random split is session-specific — paper, lighting, camera, pencil — rather than
handwriting-specific.

The first two conditions use random image-level splits, which this project treats as
**deliberately invalid demonstrations only**. They are never reported as performance
results. The project's methodological rule is participant/writer-level splitting.

---

## Reproducing the experiments

The image data is not in this repository (see *Data and ethics*). With the original
archive obtained from its source, every figure above can be regenerated:

1. Place the archive in a Drive or local folder.
2. Run the notebooks in `notebooks/handwriting/` in numerical order.
3. Compare the output against the committed reports in `docs/reports/`.

Each notebook states its purpose, inputs, method, results and limitations.
The random seed is fixed at 42 throughout.

Environment: Python 3, PyTorch, torchvision, NumPy, pandas, Pillow, scikit-learn,
matplotlib. Developed in Google Colab.

---

## Data and ethics

**No participant image data is committed to this repository, and none will be.**

The public dataset used for the pilot consists of handwriting from children aged 6–10,
collected under guardian consent for the original study. That consent does not extend to
redistribution by this project.

The project's own planned data collection involves children at a school for pupils with
special needs. That data will be more sensitive still and will never appear here.

Reproducibility is preserved through **SHA-256 fingerprints** recorded for every image in
the committed metadata. Anyone holding the original archive can verify every claim in this
repository byte for byte, without the images being republished.

Participant identifiers in any committed metadata are anonymised. No name, date of birth,
school roll number or contact detail appears in any committed file.

---

## Terminology

Participant groups are named **TR / SR** (Typical Reader / Struggling Reader) or
**HR / LR** (High Risk / Low Risk). The word "dyslexic" is not applied to any participant
without a clinical diagnosis.

This is deliberate. In the eye-tracking literature, one reference dataset's explicitly
non-diagnostic risk labels became "dyslexic" in later papers that reused the data. This
project applies the discipline to avoid contributing to that drift.

Research claims carry explicit evidence levels: CONFIRMED, SECONDARY, STRONG INFERENCE,
WEAK INFERENCE, UNKNOWN.

---

## Known limitations

| Limitation | Consequence |
|---|---|
| No participant identifiers in the public dataset | **No result in this repository is writer-independent**, and none is described as such |
| Session proxy separates recording occasions, not writers | Residual writer overlap between splits cannot be ruled out |
| Small evaluation sets (~93 test images) | Confidence intervals are wide; single-run differences of a few points are not meaningful |
| Class imbalance after cleaning | Accuracy alone is misleading; per-class metrics are required |
| Supplied `Yes`/`No` labels | Dataset labels only. They do not establish clinical validity |
| Eye-tracking prototype not runtime-tested | Camera capture and Nastaliq rendering remain unverified |

---

## Project tracking

Work is tracked in Jira (project key `DYS`). Each notebook and artefact maps to a Jira
Task, and commit messages carry the issue key.

Jira was adopted as the tracker on 26 September 2026. Work carried out before that date is
logged retrospectively from research records, with actual work periods stated in each item.
Version control adoption followed the same pattern: research outputs are committed as
documents, so the evidence is in the repository even though the commit history begins later.
This is stated openly rather than reconstructed.

---

## Branching

| Branch | Purpose |
|---|---|
| `main` | Stable, reviewable version |
| `develop` | Integration branch for completed work |
| `feature/<name>` | Feature-specific work, merged into `develop` |

---

## Team

- Noor Fatima — Product Owner & Team Lead
- Areeba Saghir — Frontend Lead
- Reham Ali — Scrum Master & Backend Lead

## Supervisor

Dr. Muhammad Farooq
Assistant Professor
Faculty of Computing and Information Technology
University of the Punjab, Lahore
