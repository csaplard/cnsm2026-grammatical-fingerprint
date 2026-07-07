"""Pipeline smoke test on synthetic data (no real data touched).

Builds 20 synthetic 'elements' as AR(1) processes with element-specific
coefficients plus a shared diurnal cycle, runs the full chain
(interpolation, z-norm, SAX, Markov fit/predict, permutation test, AAFT
surrogates, LSTM) and checks that (i) distinguishable elements are
classified well above chance, (ii) identical elements are at chance,
(iii) AAFT surrogates preserve the marginal distribution.
"""

import numpy as np

from common import interpolate_nan, znorm_params, sax_transform, windows
from models import MarkovClassifier, accuracy, permutation_test
from surrogates import aaft


def synth(n_el, T, seed, distinct=True):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    out = []
    for i in range(n_el):
        if distinct:
            phi = -0.9 + 1.8 * i / max(n_el - 1, 1)
            phase = 2 * np.pi * i / n_el
        else:
            phi, phase = 0.5, 0.0
        x = np.zeros(T)
        e = rng.standard_normal(T)
        for k in range(1, T):
            x[k] = phi * x[k - 1] + e[k]
        out.append(x + 3 * np.sin(2 * np.pi * t / 144 + phase))
    return out


def run(distinct, a=6, w=3, k=2, L=48):
    series = synth(20, 6048 + 1152, seed=1, distinct=distinct)
    train_seqs, test_wins, labels = [], [], []
    for ci, x in enumerate(series):
        xt, xa = x[:6048], x[6048:]
        mu, sd = znorm_params(xt)
        train_seqs.append(sax_transform((xt - mu) / sd, w, a))
        ww = windows(sax_transform((xa - mu) / sd, w, a), L)
        test_wins.append(ww)
        labels.append(np.full(len(ww), ci))
    clf = MarkovClassifier(order=k, alphabet=a).fit(train_seqs)
    pred = clf.predict(np.concatenate(test_wins))
    lab = np.concatenate(labels)
    acc = accuracy(pred, lab)
    p, null_mean, _ = permutation_test(pred, lab, n_perm=200, seed=2)
    return acc, p, null_mean


def main():
    acc_d, p_d, chance = run(distinct=True)
    print(f"distinct AR(1): acc={acc_d:.3f} (chance~{chance:.3f}), p={p_d:.4f}")
    assert acc_d > 3 * chance and p_d < 0.05, "distinct elements should be identifiable"

    acc_i, p_i, _ = run(distinct=False)
    print(f"identical AR(1): acc={acc_i:.3f}, p={p_i:.4f}")
    assert p_i > 0.05 or acc_i < 2 * chance, "identical elements should be near chance"

    rng = np.random.default_rng(3)
    x = np.cumsum(rng.standard_normal(4096))
    s = aaft(x, rng)
    assert np.allclose(np.sort(x), np.sort(s)), "AAFT must preserve marginal exactly"
    r_orig = np.corrcoef(x[:-1], x[1:])[0, 1]
    r_surr = np.corrcoef(s[:-1], s[1:])[0, 1]
    print(f"AAFT: lag-1 autocorr orig={r_orig:.3f} surr={r_surr:.3f} (should be similar)")

    try:
        from lstm import train_lstm, predict_lstm, make_dataset, eval_lstm
        series = synth(10, 6048 + 1152, seed=4, distinct=True)
        a_, w_, L_ = 6, 6, 48
        tr, va = [], []
        for x in series:
            xt, xa = x[:6048], x[6048:]
            mu, sd = znorm_params(xt)
            tr.append(sax_transform((xt - mu) / sd, w_, a_))
            va.append(sax_transform((xa - mu) / sd, w_, a_))
        model, vacc = train_lstm(tr, va, alphabet=a_, length=L_, seed=42)
        print(f"LSTM: val acc={vacc:.3f} (chance=0.1)")
        assert vacc > 0.3, "LSTM should beat chance on distinct AR(1)"
    except ImportError:
        print("torch not available; LSTM smoke test skipped")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
