"""Training-set-only hyperparameter sweep (PLAN.md grid, nothing beyond it).

Rolling-origin blocked validation entirely inside the training period
(2013-11-01 .. 2013-12-12, 42 days):

    fold 1: fit days 0-20,  validate days 21-27
    fold 2: fit days 0-27,  validate days 28-34
    fold 3: fit days 0-34,  validate days 35-41

Grid: alphabet a in {4, 6, 8} x PAA window w in {3, 6, 12} samples x
Markov order k in {1, 2, 3}. Selection criterion (pre-registered): mean
validation accuracy of the shallow model at L = 48 symbols.

Z-normalization statistics and interpolation are computed on the fit
segment only within each fold; validation segments are normalized with the
fit-segment statistics. The outer test windows are never touched here.

Usage: python src/sweep.py [--n-cells 50]
"""

import argparse
import itertools
import time

import numpy as np

from common import (INTERVALS_PER_DAY, RESULTS, SEED, load_channel,
                    select_cells, interpolate_nan, znorm_params,
                    sax_transform, windows, save_json)
from models import MarkovClassifier, accuracy

ALPHABETS = [4, 6, 8]
PAA_WINDOWS = [3, 6, 12]
ORDERS = [1, 2, 3]
L_SELECT = 48
L_GRID = [12, 24, 48, 96, 192]
FOLDS = [((0, 21), (21, 28)), ((0, 28), (28, 35)), ((0, 35), (35, 42))]  # days


def day_slice(d0, d1):
    return slice(d0 * INTERVALS_PER_DAY, d1 * INTERVALS_PER_DAY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=50)
    args = ap.parse_args()

    X, dates = load_channel("internet")
    cells, n_eligible = select_cells(X, args.n_cells, seed=SEED)
    print(f"{n_eligible} eligible cells; selected {len(cells)}", flush=True)

    results = []
    for a, w, k in itertools.product(ALPHABETS, PAA_WINDOWS, ORDERS):
        t0 = time.time()
        fold_acc = {L: [] for L in L_GRID}
        for (f0, f1), (v0, v1) in FOLDS:
            fit_sl, val_sl = day_slice(f0, f1), day_slice(v0, v1)
            train_seqs, val_syms = [], []
            for c in cells:
                xf = interpolate_nan(X[c, fit_sl])
                xv = interpolate_nan(X[c, val_sl])
                mu, sd = znorm_params(xf)
                train_seqs.append(sax_transform((xf - mu) / sd, w, a))
                val_syms.append(sax_transform((xv - mu) / sd, w, a))
            clf = MarkovClassifier(order=k, alphabet=a).fit(train_seqs)
            for L in L_GRID:
                if L <= k:
                    continue
                wins, labels = [], []
                for ci, sym in enumerate(val_syms):
                    ww = windows(sym, L)
                    if len(ww):
                        wins.append(ww)
                        labels.append(np.full(len(ww), ci))
                if not wins:
                    continue
                pred = clf.predict(np.concatenate(wins))
                fold_acc[L].append(accuracy(pred, np.concatenate(labels)))
        entry = {"alphabet": a, "paa_w": w, "order": k,
                 "val_acc": {str(L): (float(np.mean(v)) if v else None)
                             for L, v in fold_acc.items()},
                 "val_acc_select": (float(np.mean(fold_acc[L_SELECT]))
                                    if fold_acc[L_SELECT] else None)}
        results.append(entry)
        print(f"a={a} w={w} k={k}: acc@L48={entry['val_acc_select']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    best = max((r for r in results if r["val_acc_select"] is not None),
               key=lambda r: r["val_acc_select"])
    out = {"n_cells": int(len(cells)), "cells": [int(c) for c in cells],
           "n_eligible": int(n_eligible), "seed": SEED,
           "selection_criterion": f"mean val accuracy at L={L_SELECT}",
           "grid": results, "selected": best}
    save_json(out, RESULTS / "sweep.json")
    print("SELECTED:", best, flush=True)


if __name__ == "__main__":
    main()
