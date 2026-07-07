"""Single frozen test evaluation (run exactly once, after the freeze commit).

Reads the selected configuration from results/sweep.json (produced by the
training-set-only sweep), fits final models on the full training period,
and evaluates on Test-A (H1 confirmatory + H2 curve + controls) and, as
labeled exploratory context only, on Test-B. Everything follows PLAN.md
(commit aa73225); nothing here was tuned on test data.

Outputs: results/test_evaluation.json
"""

import json

import numpy as np

from common import (RESULTS, SEED, TRAIN, TEST_A, TEST_B, load_channel,
                    select_cells, interpolate_nan, znorm_params,
                    sax_transform, windows, save_json)
from models import MarkovClassifier, MeanLevelClassifier, accuracy, permutation_test
from surrogates import aaft

L_GRID = [12, 24, 48, 96, 192]
L_PRIMARY = 48
N_PERM = 1000
N_SURROGATES = 20
N_BOOT = 1000
N_GRID = [10, 25, 50, 100]
ALPHA_PRIOR = -0.77


def sax_all(X, cells, a, w):
    """Interpolate/normalize (train stats) and SAX all three periods."""
    train_syms, testa_syms, testb_syms, raw = [], [], [], {}
    for c in cells:
        xt = interpolate_nan(X[c, TRAIN])
        xa = interpolate_nan(X[c, TEST_A])
        xb = interpolate_nan(X[c, TEST_B])
        mu, sd = znorm_params(xt)
        train_syms.append(sax_transform((xt - mu) / sd, w, a))
        testa_syms.append(sax_transform((xa - mu) / sd, w, a))
        testb_syms.append(sax_transform((xb - mu) / sd, w, a))
        raw[c] = (xt, xa, xb)
    return train_syms, testa_syms, testb_syms, raw


def eval_windows(clf, syms, L):
    wins, labels = [], []
    for ci, sym in enumerate(syms):
        ww = windows(sym, L)
        if len(ww):
            wins.append(ww)
            labels.append(np.full(len(ww), ci))
    if not wins:
        return None, None
    return np.concatenate(wins), np.concatenate(labels)


def curve(clf, syms, l_grid, k):
    out = {}
    for L in l_grid:
        if L <= k:
            continue
        win, lab = eval_windows(clf, syms, L)
        if win is None:
            out[str(L)] = None
            continue
        pred = clf.predict(win)
        p, nm, q95 = permutation_test(pred, lab, n_perm=N_PERM, seed=SEED)
        out[str(L)] = {"acc": accuracy(pred, lab), "p": p, "null_mean": nm,
                       "null_q95": q95, "n_windows": int(len(lab))}
    return out


def transition_joint(sym, a):
    """Normalized joint distribution over (s_{t-1}, s_t) pairs."""
    counts = np.zeros((a, a))
    np.add.at(counts, (sym[:-1], sym[1:]), 1.0)
    return (counts / counts.sum()).ravel()


def jsd_bits(p, q):
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return np.sum(x[mask] * np.log2(x[mask] / y[mask]))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def h2_parameters(train_syms, a, n_cells):
    pooled = np.bincount(np.concatenate(train_syms), minlength=a).astype(float)
    pooled /= pooled.sum()
    h0 = float(-np.sum(pooled[pooled > 0] * np.log2(pooled[pooled > 0])))
    joints = [transition_joint(s, a) for s in train_syms]
    jsds = [jsd_bits(joints[i], joints[j])
            for i in range(n_cells) for j in range(i + 1, n_cells)]
    q = float(np.mean(jsds))
    b = float(np.log2(a))
    nstar_pred = h0 * b * n_cells * q ** ALPHA_PRIOR
    return {"H0_bits": h0, "B_bits": b, "C": n_cells, "Q_bits": q,
            "alpha": ALPHA_PRIOR, "Nstar_pred": float(nstar_pred)}


def crossing(l_vals, accs, band):
    """First L (interpolated) where accuracy exceeds the chance band.

    Returns (value, censored): censored='left' if already above at the
    smallest L, 'right' if never above, None otherwise.
    """
    l_vals = np.asarray(l_vals, dtype=float)
    accs = np.asarray(accs, dtype=float)
    band = np.asarray(band, dtype=float)
    above = accs > band
    if above[0]:
        return float(l_vals[0]), "left"
    if not above.any():
        return float(l_vals[-1]), "right"
    i = int(np.argmax(above))          # first True
    x0, x1 = l_vals[i - 1], l_vals[i]
    d0 = accs[i - 1] - band[i - 1]
    d1 = accs[i] - band[i]
    return float(x0 + (x1 - x0) * (-d0) / (d1 - d0)), None


def bootstrap_nstar(clf, testa_syms, l_grid, k, band_by_l, seed=SEED):
    """Bootstrap over elements: resample cells, recompute curve + crossing."""
    rng = np.random.default_rng(seed)
    n = len(testa_syms)
    per_cell = {}
    for ci, sym in enumerate(testa_syms):
        per_cell[ci] = {}
        for L in l_grid:
            if L <= k:
                continue
            ww = windows(sym, L)
            if len(ww):
                per_cell[ci][L] = clf.predict(ww) == ci
    vals, censored = [], {"left": 0, "right": 0}
    for _ in range(N_BOOT):
        pick = rng.integers(0, n, n)
        ls, accs, band = [], [], []
        for L in l_grid:
            hits = [per_cell[c][L] for c in pick if L in per_cell[c]]
            if hits:
                ls.append(L)
                accs.append(float(np.mean(np.concatenate(hits))))
                band.append(band_by_l[L])
        v, cens = crossing(ls, accs, band)
        if cens:
            censored[cens] += 1
        vals.append(v)
    vals = np.array(vals)
    return {"median": float(np.median(vals)),
            "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
            "n_left_censored": censored["left"], "n_right_censored": censored["right"]}


def main():
    sweep = json.loads((RESULTS / "sweep.json").read_text())
    sel = sweep["selected"]
    a, w, k = sel["alphabet"], sel["paa_w"], sel["order"]
    print(f"frozen config: a={a} w={w} k={k}", flush=True)

    X, dates = load_channel("internet")
    out = {"frozen_config": sel, "seed": SEED, "n_perm": N_PERM,
           "n_surrogates": N_SURROGATES, "by_n": {}}

    for n_cells in N_GRID:
        cells, n_eligible = select_cells(X, n_cells, seed=SEED)
        train_syms, testa_syms, testb_syms, raw = sax_all(X, cells, a, w)
        clf = MarkovClassifier(order=k, alphabet=a).fit(train_syms)
        res = {"cells": [int(c) for c in cells], "chance": 1.0 / n_cells}

        # H1 curve on Test-A (+ per-L permutation tests)
        res["testA_markov"] = curve(clf, testa_syms, L_GRID, k)

        # H2: prior-relation parameters + measured crossing with bootstrap CI
        res["h2_params"] = h2_parameters(train_syms, a, n_cells)
        band_by_l = {int(L): v["null_q95"]
                     for L, v in res["testA_markov"].items() if v}
        ls = sorted(band_by_l)
        accs = [res["testA_markov"][str(L)]["acc"] for L in ls]
        band = [band_by_l[L] for L in ls]
        point, cens = crossing(ls, accs, band)
        res["h2_nstar_measured"] = {"point": point, "censored": cens}
        res["h2_nstar_bootstrap"] = bootstrap_nstar(
            clf, testa_syms, ls, k, band_by_l)

        # Volume-confound baseline (raw window means, same window extents)
        mlc = MeanLevelClassifier().fit([raw[c][0] for c in cells])
        L_samp = L_PRIMARY * w
        rw, rl = [], []
        for ci, c in enumerate(cells):
            xa = raw[c][1]
            m = (len(xa) // L_samp) * L_samp
            if m:
                rw.append(xa[:m].reshape(-1, L_samp))
                rl.append(np.full(m // L_samp, ci))
        pred = mlc.predict(np.concatenate(rw))
        lab = np.concatenate(rl)
        p, nm, q95 = permutation_test(pred, lab, n_perm=N_PERM, seed=SEED)
        res["testA_meanlevel_L48"] = {"acc": accuracy(pred, lab), "p": p,
                                      "null_q95": q95}

        # Exploratory only: Test-B accuracy (context for H3, never for H1)
        res["testB_markov_exploratory"] = curve(clf, testb_syms, [L_PRIMARY], k)

        out["by_n"][str(n_cells)] = res
        print(f"N={n_cells}: acc@L48={res['testA_markov'][str(L_PRIMARY)]['acc']:.3f} "
              f"(chance={1/n_cells:.3f})", flush=True)

    # Sequence model (S1) at N=50: LSTM, architecture fixed a priori,
    # early stopping on the last training week (days 35-41, inside train).
    from common import INTERVALS_PER_DAY
    from models import permutation_test as ptest
    from lstm import train_lstm, predict_lstm
    cells50 = np.array(out["by_n"]["50"]["cells"])
    split = 35 * INTERVALS_PER_DAY
    fit_syms, es_syms, ta_syms = [], [], []
    for c in cells50:
        xt = interpolate_nan(X[c, TRAIN])
        xa = interpolate_nan(X[c, TEST_A])
        mu, sd = znorm_params(xt[:split])
        fit_syms.append(sax_transform((xt[:split] - mu) / sd, w, a))
        es_syms.append(sax_transform((xt[split:] - mu) / sd, w, a))
        ta_syms.append(sax_transform((xa - mu) / sd, w, a))
    lstm_res = {}
    for L in L_GRID:
        win, lab = eval_windows(MarkovClassifier(order=k, alphabet=a),
                                ta_syms, L)  # windowing helper only
        if win is None or min(len(s) for s in es_syms) < L:
            # structurally unavailable at the frozen w: no test window or no
            # early-stopping window of this length exists
            lstm_res[str(L)] = None
            continue
        model, val_acc = train_lstm(fit_syms, es_syms, alphabet=a, length=L,
                                    seed=SEED)
        pred = predict_lstm(model, win)
        p, nm, q95 = ptest(pred, lab, n_perm=N_PERM, seed=SEED)
        lstm_res[str(L)] = {"acc": accuracy(pred, lab), "p": p,
                            "null_q95": q95, "val_acc": val_acc,
                            "n_windows": int(len(lab))}
        print(f"LSTM L={L}: testA acc={lstm_res[str(L)]['acc']:.3f} "
              f"(val {val_acc:.3f})", flush=True)
    out["testA_lstm_N50"] = lstm_res

    # Surrogate control at N=50 (AAFT per element, per period, then frozen SAX)
    n50 = out["by_n"]["50"]
    cells = np.array(n50["cells"])
    surr_acc = []
    for r in range(N_SURROGATES):
        rng = np.random.default_rng(SEED + 1000 + r)
        tr_s, ta_s = [], []
        for c in cells:
            xt = interpolate_nan(X[c, TRAIN])
            xa = interpolate_nan(X[c, TEST_A])
            st, sa = aaft(xt, rng), aaft(xa, rng)
            mu, sd = znorm_params(st)
            tr_s.append(sax_transform((st - mu) / sd, w, a))
            ta_s.append(sax_transform((sa - mu) / sd, w, a))
        sclf = MarkovClassifier(order=k, alphabet=a).fit(tr_s)
        win, lab = eval_windows(sclf, ta_s, L_PRIMARY)
        surr_acc.append(accuracy(sclf.predict(win), lab))
    out["surrogate_control_L48_N50"] = {
        "acc_mean": float(np.mean(surr_acc)), "acc_sd": float(np.std(surr_acc)),
        "acc_all": [float(v) for v in surr_acc]}
    print(f"surrogates: acc={np.mean(surr_acc):.3f} +- {np.std(surr_acc):.3f}",
          flush=True)

    save_json(out, RESULTS / "test_evaluation.json")
    print("evaluation written", flush=True)


if __name__ == "__main__":
    main()
