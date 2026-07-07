# Audit Log — human interventions and agent decisions

Human interventions are procedural only (accept / reject / re-prompt), per the
pre-registered starting prompt. This file records them and the agent's
corresponding actions. Timestamps are local (Europe/Budapest).

| # | Date | Human action | Agent action |
|---|------|--------------|--------------|
| 1 | 2026-07-07 | Delivered starting prompt | Wrote pre-registered PLAN.md (commit `bed002b`) |
| 2 | 2026-07-07 | **Rejected: plan incomplete** (H2 parameter mapping missing; H1/H3 confounded) | Revised plan: pre-registered H₀/B/C/Q/α mapping and consistency criterion; split held-out period into Test-A (H1, 2013-12-13..20) and Test-B (H3, 2013-12-21..end) (commit `aa73225`) |
| 3 | 2026-07-07 | **Accepted**, conditional on clarification: AAFT surrogates must be generated from raw series, then SAX-discretized with frozen parameters | Confirmed (this was the design as written in PLAN.md). Proceeded: data acquisition, then training-set-only sweep. |
| 4 | 2026-07-07 | Authorized guestbook email (procedural; dataset requires an email per download) | Used csaplar.d@gmail.com; downloaded and verified all 62 files. |
| 5 | 2026-07-07 | **Accepted** unverifiable prior-relation handling (keep as pre-registered proposal; documented deviation from the citation instruction stays in Limitations; human supplies no citations per pre-registration). Will provide a public GitHub URL as procedural data for the agent to push history and insert into the Disclosure Statement. | No citations added. Awaiting repo URL to push full history, verify root commit + freeze commit 760de0c on the remote, insert URL, and build the final camera-ready PDF. |

## Final results summary (single test evaluation, frozen config a=8/w=12/k=2)

- **H1 (fingerprint): supported.** Test-A Markov accuracy far exceeds chance at
  all N and L (N=50, L=48: 0.51 vs chance 0.02, permutation p<0.001).
- **Surrogate control:** AAFT accuracy 0.139±0.034 — most identifiability is
  temporal/grammatical, but a ~7×-chance marginal component remains.
- **Volume confound:** mean-level baseline 0.46 (N=50), 0.85 (N=10) — an
  independent, at small N stronger, fingerprint. The symbolic result is
  volume-independent (post-normalization).
- **H2 (scaling): inconsistent.** Measured N*<12 symbols vs predicted
  400–2794; reported as inconsistency, mapping not adjusted.
- **S1:** shallow Markov (0.51) > LSTM (0.21); sequence model unnecessary.
- **H3 (drift): supported.** Symbolic drift leads raw drift by median 54 h
  (95% CI 18–90 h).

Deliverables: paper/main.pdf (IEEE, 4 pp, ≤5 limit), SUMMARY.md (plain-language),
this repository. All 6 references verified real; prior scaling relation could
not be located and is treated as a pre-registered proposal, not a citation.

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

## Test-evaluation rerun note (transparency)

The first run of src/evaluate.py after the freeze commit crashed at LSTM
L=96 (the early-stopping segment, 84 symbols at the frozen w=12, cannot
contain a 96-symbol window — same structural limitation already noted at
the freeze). The fix skips structurally unavailable L values; **no model,
hyperparameter, or selection was changed after seeing test results.** The
script was rerun once to completion; Markov test numbers were identical
across both runs (deterministic pipeline, seed 42).

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
