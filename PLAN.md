# Experiment Plan — "Do Network Elements Have a Grammatical Fingerprint?"

*(Pre-registered plan, written by the autonomous research agent per the starting prompt. Revision 2: adds the pre-registered H2 parameter mapping and separates the H1 and H3 evaluation windows, per rejection feedback. No data has been downloaded and no experiment has been run at the time of this commit. Awaiting human approval before proceeding.)*

## Dataset

**Telecom Italia Big Data Challenge — Milano telecommunications grid** (Barlacchi et al., *Scientific Data* 2, 150055, 2015). Per-cell activity (SMS-in/out, call-in/out, internet) over a 100×100 spatial grid of Milan, 10-minute resolution, 62 days (2013-11-01 → 2014-01-01). Source: Harvard Dataverse, DOI [10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV) (grid geometry: DOI 10.7910/DVN/QJWLFU), license ODbL (to be verified at download and recorded). Download via the Dataverse API; the acquisition script, file checksums, and exact version will be committed to this repository. The dataset satisfies preference (a) of the method constraints; each grid cell is one "network element." The **internet activity** channel is the primary signal (densest channel); ≈ 8,928 samples per element.

The dataset contains a calendar-documented regime change (Christmas/New Year), so **H3 is retained**.

## Elements (N)

Primary study: **N = 50 cells**, selected by stratified sampling over deciles of training-period mean internet activity (5 cells per decile, fixed seed 42), to avoid cherry-picking high-traffic cells. Secondary scaling analysis: N ∈ {10, 25, 50, 100} with the same stratified procedure. Cells with > 5 % missing intervals in the training period are excluded before sampling (exclusion count reported).

## Temporal split — two disjoint test windows (H1/H3 separation)

Train: 2013-11-01 → 2013-12-12 (42 days). The held-out remainder is split into two disjoint, pre-registered windows with a **fixed calendar boundary that will not be moved after seeing the data**:

- **Test-A (H1, confirmatory): 2013-12-13 00:00 → 2013-12-20 23:59** (8 days). This window precedes the Italian school-holiday/Christmas period; it is the *same-regime* window. **All confirmatory H1 statistics — headline accuracy, permutation test, surrogate control, volume-confound baseline — and the H2 accuracy-vs-L curve and N\* estimate are computed on Test-A only.** A fingerprint that persists from the training period into Test-A satisfies the pre-registered stability definition.
- **Test-B (H3, drift): 2013-12-21 00:00 → 2014-01-01 (end of data)** (12 days). This window contains the calendar-documented regime change (Christmas/New Year). It is used **only** for the H3 drift analysis. Classification accuracy on Test-B is reported as *exploratory context* (labeled as such) and plays no role in accepting or rejecting H1.

This removes the confound: a low accuracy on Test-A means "no stable fingerprint" (H1 fails); a fingerprint that holds on Test-A but drifts inside Test-B is evidence *for* H3, not against H1. The Dec 20/21 boundary is chosen because Italian school holidays and the associated mobility change begin in the days immediately before Christmas; it is fixed here, before any data inspection.

No shuffling across time anywhere. All model and parameter selection uses blocked (temporal) validation *within the training period only*; each test window is evaluated exactly once after a freeze commit. Window counts per sequence length L will be reported (at the longest L, Test-A supports few non-overlapping windows per element; uncertainty is handled by bootstrap over elements).

## Discretization (SAX) — sweep grid

Per-element z-normalization using **training-period statistics only** (volume-confound mitigation), then PAA + SAX:

- Alphabet size **a ∈ {4, 6, 8}**
- PAA segment length **w ∈ {3, 6, 12}** raw samples per symbol (= 30, 60, 120 min)

Full 3×3 grid swept in training-set validation; the final (a, w) is selected on validation accuracy of the shallow model and justified in the paper; sensitivity across the grid is reported. No parameters outside this grid will be explored (honest-null clause).

## Classification sequence lengths

Test sequences of **L ∈ {12, 24, 48, 96, 192} symbols** (non-overlapping windows, Test-A). H1 is tested at the mid-grid length L = 48; the accuracy-vs-L curve over all L values feeds the H2 chance-crossing estimate N\* (bootstrap CI, 1,000 resamples).

## H2 — pre-registered mapping to the prior scaling relation

The prior proposal is N\* = H₀·B·C·Q^α with α ≈ −0.77 (the authors' own earlier proposal from quantum-circuit noise and EEG work; it will be cited as such and explicitly flagged as not an established law). To prevent post-hoc fitting, the mapping of every parameter to a quantity measurable in *this* study is fixed now, before any data is downloaded. All four are computed **on the training set only**:

- **N\*** (measured): the number of symbols per test sequence at which the shallow classifier's accuracy-vs-L curve crosses the permutation-derived chance band (interpolated crossing; bootstrap CI over elements, 1,000 resamples), on Test-A.
- **H₀**: order-0 (marginal) Shannon entropy, in bits per symbol, of the pooled symbol distribution over all N elements' training sequences under the final (a, w).
- **B**: symbol information capacity, log₂(a) bits, for the final alphabet size a.
- **C**: the number of classes, C = N (number of elements; 50 in the primary study).
- **Q**: source separability, defined as the mean pairwise Jensen–Shannon divergence (in bits) between the per-element order-1 symbol-transition distributions, estimated on the training set.
- **α**: fixed at −0.77 (the prior point estimate); it is **not** re-fitted.

Comparison procedure (also fixed now): compute the predicted N\*_pred = H₀·B·C·Q^(−0.77) and report the ratio N\*_meas / N\*_pred with the bootstrap CI of N\*_meas propagated through. **Consistency** is declared iff N\*_pred lies within the 95 % CI of N\*_meas; otherwise inconsistency is reported, with the magnitude of the discrepancy. Both outcomes go into the paper with equal prominence. The relation is treated as an order-of-magnitude proposal, never as a law. If, when the prior publications are retrieved for Related Work, their published parameter definitions turn out to differ from this operational mapping, the paper reports the discrepancy as a limitation of the comparison — the mapping above is **not** revised to improve agreement, and no alternative mapping is evaluated.

## Models

1. **Shallow baseline (mandatory):** Markov / n-gram classifier, orders 1–3, additive smoothing; class = argmax per-element sequence log-likelihood.
2. **Sequence model:** single-layer **LSTM** (~64 hidden units, symbol embedding), chosen over a transformer because per-class data is modest and an LSTM is the more conservative, lower-variance choice at this scale. Identical splits and sequences as (1).
3. **Confound baseline:** mean-level-only classifier (classifies from the per-window mean of the *un-normalized* series) to quantify how much identification is achievable from traffic volume alone.

## Controls

- **Permutation test:** ≥ 1,000 label permutations for the chance baseline of every headline accuracy (p-values reported).
- **Surrogate control:** AAFT surrogates generated per element from the continuous series, then identically SAX-discretized and classified; if accuracy survives, the grammar claim is weakened accordingly in the paper.
- **Volume confound:** per-element z-normalization (above) plus the mean-level-only baseline (model 3).

## H3 drift analysis

Conducted on **Test-B only** (the drift window; the model and its reference statistics are frozen from the training period, and Test-A serves as the same-regime sanity window). Identical change-point detector (CUSUM; Page–Hinkley as robustness check) applied to (a) drift of the symbolic signal — per-element classifier confidence and symbol-transition-distribution divergence (sliding-window Jensen–Shannon against the training-period reference) — and (b) drift of raw first-order metrics (sliding mean, variance) on the same data. Lead time = detection timestamp of (b) minus detection timestamp of (a), reported in hours with bootstrap CI over elements; detector thresholds are calibrated on the training period identically for both signals so neither is favored.

## Reproducibility

Single repository; Python; fixed seeds (global seed 42, per-stage seeds derived and stated in the paper); one command (`python reproduce.py`) regenerates every figure and table. Hyperparameter freeze documented by commit hash before the single test-set evaluation.

---

**Status: awaiting explicit human approval before any data download or experiment execution.**
