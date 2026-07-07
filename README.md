# Do Network Elements Have a Grammatical Fingerprint?

Autonomous, pre-registered empirical study for the **CNSM 2026 AI-Generated
Paper Track**. All technical content (design, code, analysis, figures, text)
was produced by an autonomous agent; the human made procedural
accept/reject/re-prompt decisions only. See [LOG.md](LOG.md) for the full
intervention audit trail and [`cnsm2026_research_director_prompt.md`](cnsm2026_research_director_prompt.md)
for the verbatim starting prompt (the complete human technical contribution).

## Result in one line
A shallow Markov model over SAX-symbolized cellular telemetry identifies the
source cell far above chance (0.51 vs 0.02 at N=50, permutation p<0.001); the
signal is largely temporal (grammatical) and volume-independent, but coexists
with a competing volume fingerprint, needs very little data (N\*<12 symbols,
inconsistent with the prior scaling proposal), and drifts ~54 h before
first-order metrics during a regime change.

## Reproduce everything
```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python reproduce.py --email YOUR@EMAIL   # email is required by the dataset's Dataverse guestbook
```
This downloads and MD5-verifies the dataset, runs the training-only sweep, the
single frozen test evaluation, the H3 drift analysis, and regenerates every
figure and table. Global seed: 42.

## Pipeline (`src/`)
| file | role |
|------|------|
| `acquire.py` | download + MD5-verify + reduce the Milano dataset (guestbook-gated) |
| `common.py` | pre-registered splits, stratified cell selection, SAX, normalization |
| `models.py` | order-k Markov baseline, mean-level confound baseline, permutation test |
| `surrogates.py` | AAFT surrogates from the raw series |
| `lstm.py` | single-layer LSTM sequence model (S1) |
| `sweep.py` | training-set-only rolling-origin hyperparameter sweep |
| `evaluate.py` | single frozen Test-A evaluation (H1/H2/controls/LSTM) + exploratory Test-B |
| `h3_drift.py` | CUSUM / Page–Hinkley drift lead-time analysis (H3) |
| `figures.py` | all paper figures and tables from `results/*.json` |

## Key commits (audit trail)
- Pre-registered plan: see `PLAN.md` history (rejected once, revised).
- **Freeze commit** `760de0c` — model selection frozen *before* the single
  test-set evaluation, per the test-hygiene rule.

## Deliverables
- Paper: [`paper/main.tex`](paper/main.tex) → `paper/main.pdf` (IEEE 2-column, 4 pp).
- Plain-language summary: [SUMMARY.md](SUMMARY.md).
- This repository: code, data-acquisition script, seeds, provenance manifest,
  transcript/commit history.

## Data & license
Telecom Italia Big Data Challenge, Milano grid (Barlacchi et al., *Scientific
Data* 2015), Harvard Dataverse `doi:10.7910/DVN/EGZHFV` v1.3, **ODbL 1.0**.
Raw files are not redistributed here; `data/manifest.json` records the verified
checksums. Code in this repository is provided for reproduction of the study.
