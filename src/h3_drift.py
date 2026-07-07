"""H3: does symbolic-grammar drift lead raw first-order drift? (Test-B)

Pre-registered design (PLAN.md): identical change-point detection (CUSUM;
Page-Hinkley robustness check) on (a) symbolic drift signals — per-element
Markov log-likelihood under the element's own frozen model, and sliding
Jensen-Shannon divergence of the order-1 transition joint against the
training reference — and (b) raw first-order signals — sliding mean and
variance of the same series. Thresholds calibrated on the training period
identically for all signals: alarm when the CUSUM statistic exceeds the
maximum value it attains anywhere in the training period.

Sliding window: 24 h (integrates the diurnal cycle; weekly variation is
absorbed into the training-period calibration), stride 6 h. Each signal is
standardized per element by the mean/sd of its training-window values,
then |z| is averaged over elements to give one detector input per signal.

Lead time = t_alarm(raw) - t_alarm(symbolic), in hours; bootstrap CI over
elements (1000 resamples).

Outputs: results/h3_drift.json
"""

import json

import numpy as np

from common import (INTERVALS_PER_DAY, RESULTS, SEED, TRAIN, TEST_A, TEST_B,
                    load_channel, interpolate_nan, znorm_params,
                    sax_transform, save_json)
from models import MarkovClassifier
from evaluate import transition_joint, jsd_bits

WIN_H = 24          # sliding window, hours
STRIDE_H = 6        # stride, hours
SAMPLES_PER_H = 6   # 10-min intervals
N_BOOT = 1000


def sliding_stats(x_raw, sym, clf_logp, train_joint, a, w, k):
    """Per-window signals for one element.

    x_raw: raw samples of the period; sym: its SAX symbols (train stats).
    Returns dict of signal name -> array over windows, plus window count.
    """
    win_s = WIN_H * SAMPLES_PER_H
    stride_s = STRIDE_H * SAMPLES_PER_H
    win_y = win_s // w                      # symbols per window
    stride_y = stride_s // w
    n_win = (len(x_raw) - win_s) // stride_s + 1
    sig = {"raw_mean": np.empty(n_win), "raw_var": np.empty(n_win),
           "sym_loglik": np.empty(n_win), "sym_jsd": np.empty(n_win)}
    for i in range(n_win):
        xs = x_raw[i * stride_s: i * stride_s + win_s]
        ys = sym[i * stride_y: i * stride_y + win_y]
        sig["raw_mean"][i] = xs.mean()
        sig["raw_var"][i] = xs.var()
        ctx = np.zeros(len(ys) - k, dtype=np.int64)
        for j in range(1, k + 1):
            ctx += ys[k - j: len(ys) - j] * (a ** (j - 1))
        sig["sym_loglik"][i] = clf_logp[ctx, ys[k:]].mean()
        sig["sym_jsd"][i] = jsd_bits(transition_joint(ys, a), train_joint)
    return sig


def cusum(z, drift=0.5):
    """One-sided CUSUM on |z| (deviation magnitude)."""
    s = np.zeros(len(z))
    for i in range(1, len(z)):
        s[i] = max(0.0, s[i - 1] + z[i] - drift)
    return s


def page_hinkley(z, delta=0.05):
    m, ph = 0.0, np.zeros(len(z))
    mn = 0.0
    cum = 0.0
    for i in range(len(z)):
        cum += z[i] - delta
        mn = min(mn, cum)
        ph[i] = cum - mn
    return ph


def detect(signals_train, signals_test, detector):
    """Calibrate on train, alarm index on test (None if no alarm)."""
    out = {}
    for name in signals_train:
        thr = detector(signals_train[name]).max()
        s = detector(signals_test[name])
        idx = np.where(s > thr)[0]
        out[name] = (int(idx[0]) if len(idx) else None, float(thr))
    return out


def aggregate(per_el, names):
    """Standardize per element by its training windows, mean |z| over elements."""
    agg_tr, agg_te = {}, {}
    for nm in names:
        ztr, zte = [], []
        for el in per_el:
            tr, te = el[nm]["train"], el[nm]["test"]
            mu, sd = tr.mean(), max(tr.std(), 1e-12)
            ztr.append(np.abs((tr - mu) / sd))
            zte.append(np.abs((te - mu) / sd))
        agg_tr[nm] = np.mean(ztr, axis=0)
        agg_te[nm] = np.mean(zte, axis=0)
    return agg_tr, agg_te


def main():
    sweep = json.loads((RESULTS / "sweep.json").read_text())
    sel = sweep["selected"]
    a, w, k = sel["alphabet"], sel["paa_w"], sel["order"]
    cells = json.loads((RESULTS / "test_evaluation.json").read_text())["by_n"]["50"]["cells"]

    X, dates = load_channel("internet")
    names = ["raw_mean", "raw_var", "sym_loglik", "sym_jsd"]
    per_el = []
    for c in cells:
        xt = interpolate_nan(X[c, TRAIN])
        xab = interpolate_nan(np.concatenate([X[c, TEST_A], X[c, TEST_B]]))
        mu, sd = znorm_params(xt)
        zt, zab = (xt - mu) / sd, (xab - mu) / sd
        st = sax_transform(zt, w, a)
        sab = sax_transform(zab, w, a)
        clf = MarkovClassifier(order=k, alphabet=a).fit([st])
        tj = transition_joint(st, a)
        sig_tr = sliding_stats(zt, st, clf.logp[0], tj, a, w, k)
        sig_te = sliding_stats(zab, sab, clf.logp[0], tj, a, w, k)
        per_el.append({nm: {"train": sig_tr[nm], "test": sig_te[nm]} for nm in names})
    agg_tr, agg_te = aggregate(per_el, names)

    result = {"frozen_config": sel, "window_h": WIN_H, "stride_h": STRIDE_H,
              "test_period_start": "2013-12-13T00:00 local", "detectors": {}}
    for det_name, det in [("cusum", cusum), ("page_hinkley", page_hinkley)]:
        d = detect(agg_tr, agg_te, det)
        result["detectors"][det_name] = {
            nm: {"alarm_window": d[nm][0], "threshold": d[nm][1],
                 "alarm_hours_after_dec13": (d[nm][0] * STRIDE_H + WIN_H
                                             if d[nm][0] is not None else None)}
            for nm in names}

    # Bootstrap lead time over elements (CUSUM, best symbolic vs best raw:
    # pre-registered comparison is earliest symbolic alarm vs earliest raw alarm)
    rng = np.random.default_rng(SEED)
    leads, sym_al, raw_al = [], [], []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(per_el), len(per_el))
        btr, bte = aggregate([per_el[i] for i in pick], names)
        d = detect(btr, bte, cusum)
        sy = [d[nm][0] for nm in ("sym_loglik", "sym_jsd") if d[nm][0] is not None]
        ra = [d[nm][0] for nm in ("raw_mean", "raw_var") if d[nm][0] is not None]
        if sy and ra:
            lead = (min(ra) - min(sy)) * STRIDE_H
            leads.append(lead)
        sym_al.append(min(sy) if sy else None)
        raw_al.append(min(ra) if ra else None)
    if leads:
        leads = np.array(leads, dtype=float)
        result["lead_time_hours"] = {
            "n_boot_with_both_alarms": int(len(leads)),
            "median": float(np.median(leads)),
            "ci95": [float(np.quantile(leads, 0.025)),
                     float(np.quantile(leads, 0.975))]}
    else:
        result["lead_time_hours"] = {"n_boot_with_both_alarms": 0}

    save_json(result, RESULTS / "h3_drift.json")
    print(json.dumps(result["detectors"]["cusum"], indent=1))
    print("lead:", result["lead_time_hours"])


if __name__ == "__main__":
    main()
