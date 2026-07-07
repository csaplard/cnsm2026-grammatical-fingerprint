"""Classifiers for symbolic sequences.

MarkovClassifier is the mandatory shallow baseline (PLAN.md): per-element
order-k transition tables with additive smoothing; a test window is assigned
to the element maximizing the conditional log-likelihood. MeanLevelClassifier
is the volume-confound baseline: it sees only the per-window mean of the
un-normalized series.
"""

import numpy as np


class MarkovClassifier:
    def __init__(self, order, alphabet, smoothing=1.0):
        self.k = order
        self.a = alphabet
        self.lam = smoothing
        self.logp = None          # [n_classes, a^k, a]

    def _contexts(self, sym):
        """Encode order-k contexts; returns (ctx, nxt) for t >= k."""
        k, a = self.k, self.a
        nxt = sym[k:]
        ctx = np.zeros(len(sym) - k, dtype=np.int64)
        for j in range(1, k + 1):
            ctx += sym[k - j: len(sym) - j] * (a ** (j - 1))
        return ctx, nxt

    def fit(self, train_seqs):
        """train_seqs: list (one per class) of 1-D symbol arrays."""
        a, k = self.a, self.k
        n_ctx = a ** k
        self.logp = np.empty((len(train_seqs), n_ctx, a))
        for c, sym in enumerate(train_seqs):
            counts = np.zeros((n_ctx, a))
            ctx, nxt = self._contexts(sym)
            np.add.at(counts, (ctx, nxt), 1.0)
            counts += self.lam
            self.logp[c] = np.log(counts / counts.sum(axis=1, keepdims=True))
        return self

    def scores(self, win):
        """win: [n_windows, L] symbol matrix -> [n_windows, n_classes] log-lik."""
        ctxs, nxts = [], []
        for w in win:
            ctx, nxt = self._contexts(w)
            ctxs.append(ctx)
            nxts.append(nxt)
        ctx = np.stack(ctxs)
        nxt = np.stack(nxts)
        # [n_classes, n_windows, L-k] gathered log-probs, summed over time
        return self.logp[:, ctx, nxt].sum(axis=2).T

    def predict(self, win):
        return self.scores(win).argmax(axis=1)


class MeanLevelClassifier:
    """Volume-confound baseline: nearest training mean on window means."""

    def fit(self, train_raw):
        """train_raw: list (one per class) of 1-D raw (un-normalized) series."""
        self.means = np.array([np.nanmean(x) for x in train_raw])
        return self

    def predict(self, win_raw):
        """win_raw: [n_windows, L_samples] raw windows."""
        m = np.nanmean(win_raw, axis=1)
        return np.abs(m[:, None] - self.means[None, :]).argmin(axis=1)


def accuracy(pred, labels):
    return float(np.mean(pred == labels))


def permutation_test(pred, labels, n_perm=1000, seed=42):
    """Label-permutation test for classification accuracy.

    Permutes the true labels of the test windows and recomputes accuracy,
    giving the null distribution of accuracy under no association.
    Returns (p_value, null_mean, null_q95).
    """
    rng = np.random.default_rng(seed)
    obs = accuracy(pred, labels)
    null = np.empty(n_perm)
    lab = labels.copy()
    for i in range(n_perm):
        rng.shuffle(lab)
        null[i] = accuracy(pred, lab)
    p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return float(p), float(null.mean()), float(np.quantile(null, 0.95))
