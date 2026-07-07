# Audit Log — human interventions and agent decisions

Human interventions are procedural only (accept / reject / re-prompt), per the
pre-registered starting prompt. This file records them and the agent's
corresponding actions. Timestamps are local (Europe/Budapest).

| # | Date | Human action | Agent action |
|---|------|--------------|--------------|
| 1 | 2026-07-07 | Delivered starting prompt | Wrote pre-registered PLAN.md (commit `bed002b`) |
| 2 | 2026-07-07 | **Rejected: plan incomplete** (H2 parameter mapping missing; H1/H3 confounded) | Revised plan: pre-registered H₀/B/C/Q/α mapping and consistency criterion; split held-out period into Test-A (H1, 2013-12-13..20) and Test-B (H3, 2013-12-21..end) (commit `aa73225`) |
| 3 | 2026-07-07 | **Accepted**, conditional on clarification: AAFT surrogates must be generated from raw series, then SAX-discretized with frozen parameters | Confirmed (this was the design as written in PLAN.md). Proceeding: data acquisition, then training-set-only sweep. |

## Resolved procedural question (guestbook)

The Dataverse dataset is gated by a guestbook ("Privacy risk assessment",
guestbook id 96) that requires an **email address** with each file download
(no other fields, no custom questions). The agent paused and asked; the
human authorized using csaplar.d@gmail.com (intervention #4, 2026-07-07,
procedural — no technical content supplied). Download proceeded with that
guestbook response.

## Hyperparameter freeze (constraint 6)

Training-set-only sweep (rolling-origin, 3 folds, PLAN.md grid) selected
**alphabet a=8, PAA window w=12 samples (120 min), Markov order k=2**
(mean validation accuracy 0.4533 at L=48, N=50, chance 0.02). LSTM
architecture was fixed a priori (src/lstm.py); its only tuned setting is
early stopping inside the training period. **All model selection is hereby
frozen at this commit.** The next run of src/evaluate.py is the single
test evaluation permitted by the plan; src/h3_drift.py is likewise run
once. Note: at w=12, L∈{96} yields one window per element on Test-A and
L=192 yields none; this is a structural consequence of the frozen choice
and is reported as such.

## Dataset provenance (recorded before download)

- Dataset: "Telecommunications - SMS, Call, Internet - MI", Harvard Dataverse,
  DOI 10.7910/DVN/EGZHFV, **version 1.3 (RELEASED)**, queried via the Dataverse
  native API on 2026-07-07.
- License: **ODbL 1.0** (per the dataset's terms of use).
- 62 daily tab-separated files, 2013-11-01 … 2014-01-01, total 20,804,803,507
  bytes. Per-file IDs, sizes and MD5 checksums are recorded in
  `data/manifest.json` by `src/acquire.py`; raw files are deleted after
  MD5 verification and reduction (per-cell aggregation over country codes),
  so only the reduced matrices (`data/daily/*.npz`, gitignored) and the
  manifest remain on disk.
