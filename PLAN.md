# Experiment Plan — "Do Network Elements Have a Grammatical Fingerprint?"

*(Pre-registered plan, written by the autonomous research agent per the starting prompt. No data has been downloaded and no experiment has been run at the time of this commit. Awaiting human approval before proceeding.)*

## Dataset

**Telecom Italia Big Data Challenge — Milano telecommunications grid** (Barlacchi et al., *Scientific Data* 2, 150055, 2015). Per-cell activity (SMS-in/out, call-in/out, internet) over a 100×100 spatial grid of Milan, 10-minute resolution, 62 days (2013-11-01 → 2014-01-01). Source: Harvard Dataverse, DOI [10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV) (grid geometry: DOI 10.7910/DVN/QJWLFU), license ODbL (to be verified at download and recorded). Download via the Dataverse API; the acquisition script, file checksums, and exact version will be committed to this repository. The dataset satisfies preference (a) of the method constraints; each grid cell is one "network element." The **internet activity** channel is the primary signal (densest channel); ≈ 8,928 samples per element.

The dataset contains a calendar-documented regime change (Christmas/New Year), so **H3 is retained**.

## Elements (N)

Primary study: **N = 50 cells**, selected by stratified sampling over deciles of training-period mean internet activity (5 cells per decile, fixed seed 42), to avoid cherry-picking high-traffic cells. Secondary scaling analysis: N ∈ {10, 25, 50, 100} with the same stratified procedure. Cells with > 5 % missing intervals in the training period are excluded before sampling (exclusion count reported).

## Temporal split

Train: 2013-11-01 → 2013-12-12 (42 days, ≈ 68 %). Test: 2013-12-13 → 2014-01-01 (20 days, ≈ 32 %; contains the Christmas regime change used by H3). No shuffling across time. All model and parameter selection uses blocked (temporal) validation *within the training period only*; the test set is evaluated exactly once after a freeze commit.

## Discretization (SAX) — sweep grid

Per-element z-normalization using **training-period statistics only** (volume-confound mitigation), then PAA + SAX:

- Alphabet size **a ∈ {4, 6, 8}**
- PAA segment length **w ∈ {3, 6, 12}** raw samples per symbol (= 30, 60, 120 min)

Full 3×3 grid swept in training-set validation; the final (a, w) is selected on validation accuracy of the shallow model and justified in the paper; sensitivity across the grid is reported. No parameters outside this grid will be explored (honest-null clause).

## Classification sequence lengths

Test sequences of **L ∈ {12, 24, 48, 96, 192} symbols** (non-overlapping windows). H1 is tested at the mid-grid length L = 48; the accuracy-vs-L curve over all L values feeds the H2 chance-crossing estimate N\* (bootstrap CI, 1,000 resamples), compared against the prior proposal N\* = H₀·B·C·Q^α (α ≈ −0.77), reported as consistent or inconsistent with CIs either way.

## Models

1. **Shallow baseline (mandatory):** Markov / n-gram classifier, orders 1–3, additive smoothing; class = argmax per-element sequence log-likelihood.
2. **Sequence model:** single-layer **LSTM** (~64 hidden units, symbol embedding), chosen over a transformer because per-class data is modest and an LSTM is the more conservative, lower-variance choice at this scale. Identical splits and sequences as (1).
3. **Confound baseline:** mean-level-only classifier (classifies from the per-window mean of the *un-normalized* series) to quantify how much identification is achievable from traffic volume alone.

## Controls

- **Permutation test:** ≥ 1,000 label permutations for the chance baseline of every headline accuracy (p-values reported).
- **Surrogate control:** AAFT surrogates generated per element from the continuous series, then identically SAX-discretized and classified; if accuracy survives, the grammar claim is weakened accordingly in the paper.
- **Volume confound:** per-element z-normalization (above) plus the mean-level-only baseline (model 3).

## H3 drift analysis

Identical change-point detector (CUSUM; Page–Hinkley as robustness check) applied to (a) drift of the symbolic signal — per-element classifier confidence and symbol-transition-distribution divergence (sliding-window Jensen–Shannon) — and (b) drift of raw first-order metrics (sliding mean, variance) on the same data. Lead time of (a) vs (b) around the documented regime change reported in hours with bootstrap CI.

## Reproducibility

Single repository; Python; fixed seeds (global seed 42, per-stage seeds derived and stated in the paper); one command (`python reproduce.py`) regenerates every figure and table. Hyperparameter freeze documented by commit hash before the single test-set evaluation.

---

**Status: awaiting explicit human approval before any data download or experiment execution.**
