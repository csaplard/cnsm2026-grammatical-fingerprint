"""Shared data handling for the grammatical-fingerprint study.

All period boundaries, selection rules, and normalization discipline follow
the pre-registered PLAN.md (commit aa73225). Global seed: 42.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
DAILY_DIR = ROOT / "data" / "daily"
RESULTS = ROOT / "results"

SEED = 42
INTERVALS_PER_DAY = 144
N_DAYS = 62                      # 2013-11-01 .. 2014-01-01

# Pre-registered period boundaries (day indices, then sample indices).
TRAIN_DAYS = (0, 42)             # 2013-11-01 .. 2013-12-12
TEST_A_DAYS = (42, 50)           # 2013-12-13 .. 2013-12-20  (H1, confirmatory)
TEST_B_DAYS = (50, 62)           # 2013-12-21 .. 2014-01-01  (H3, drift only)
TRAIN = slice(TRAIN_DAYS[0] * INTERVALS_PER_DAY, TRAIN_DAYS[1] * INTERVALS_PER_DAY)
TEST_A = slice(TEST_A_DAYS[0] * INTERVALS_PER_DAY, TEST_A_DAYS[1] * INTERVALS_PER_DAY)
TEST_B = slice(TEST_B_DAYS[0] * INTERVALS_PER_DAY, TEST_B_DAYS[1] * INTERVALS_PER_DAY)

MAX_MISSING_FRAC = 0.05          # eligibility threshold on the training period


def load_channel(channel="internet"):
    """Concatenate daily matrices -> (X [10000 x 8928], dates)."""
    files = sorted(DAILY_DIR.glob("*.npz"))
    if len(files) != N_DAYS:
        raise RuntimeError(f"expected {N_DAYS} daily files, found {len(files)}")
    mats, dates = [], []
    for f in files:
        z = np.load(f)
        n = int(z["n_intervals"])
        if n != INTERVALS_PER_DAY:
            raise RuntimeError(f"{f.name}: {n} intervals, expected {INTERVALS_PER_DAY}")
        mats.append(z[channel])
        dates.append(f.stem)
    return np.concatenate(mats, axis=1), dates


def select_cells(X, n_cells, seed=SEED):
    """Stratified selection over deciles of training-period mean activity.

    Eligibility: <= 5% missing intervals in the training period. Cells are
    ranked by training-period mean, split into 10 deciles, and sampled
    uniformly within deciles (floor allocation, remainder assigned to
    seeded-random distinct deciles). Returns sorted cell indices (0-based).
    """
    tr = X[:, TRAIN]
    missing = np.isnan(tr).mean(axis=1)
    eligible = np.where(missing <= MAX_MISSING_FRAC)[0]
    means = np.nanmean(tr[eligible], axis=1)
    order = eligible[np.argsort(means, kind="stable")]
    deciles = np.array_split(order, 10)
    rng = np.random.default_rng(seed)
    base, rem = divmod(n_cells, 10)
    extra = np.zeros(10, dtype=int)
    extra[rng.choice(10, size=rem, replace=False)] = 1
    chosen = []
    for d, dec in enumerate(deciles):
        k = base + extra[d]
        chosen.extend(rng.choice(dec, size=k, replace=False))
    return np.sort(np.array(chosen)), len(eligible)


def interpolate_nan(x):
    """Linear interpolation over NaN gaps within a single period (1-D)."""
    x = x.astype(np.float64).copy()
    nans = np.isnan(x)
    if nans.all():
        raise ValueError("all-NaN series")
    if nans.any():
        idx = np.arange(len(x))
        x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x


def znorm_params(x_train):
    mu = np.mean(x_train)
    sd = np.std(x_train)
    return mu, (sd if sd > 1e-12 else 1.0)


def sax_breakpoints(alphabet):
    return norm.ppf(np.linspace(0, 1, alphabet + 1)[1:-1])


def sax_transform(x, paa_w, alphabet):
    """PAA over non-overlapping blocks of paa_w samples, then SAX symbols."""
    n = (len(x) // paa_w) * paa_w
    paa = x[:n].reshape(-1, paa_w).mean(axis=1)
    return np.searchsorted(sax_breakpoints(alphabet), paa).astype(np.int64)


def windows(sym, length):
    """Non-overlapping windows of `length` symbols; drops the remainder."""
    k = len(sym) // length
    return sym[: k * length].reshape(k, length)


def prepare_period(X, cells, period, mu_sd=None):
    """Interpolate + z-normalize one period for the selected cells.

    mu_sd: dict cell -> (mu, sd). If None, computed from this period
    (only valid when the period IS the fitting period).
    Returns (Z [len(cells) x T_period], mu_sd).
    """
    out = np.empty((len(cells), period.stop - period.start))
    params = {} if mu_sd is None else mu_sd
    for i, c in enumerate(cells):
        x = interpolate_nan(X[c, period])
        if mu_sd is None:
            params[c] = znorm_params(x)
        mu, sd = params[c]
        out[i] = (x - mu) / sd
    return out, params


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
