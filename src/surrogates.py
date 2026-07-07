"""Amplitude-adjusted Fourier transform (AAFT) surrogates.

Generated from the raw continuous per-element series (never from symbol
sequences), then discretized with the frozen SAX parameters — per PLAN.md
and the human's acceptance condition (LOG.md #3).

AAFT preserves the marginal amplitude distribution exactly and the linear
power spectrum approximately, while destroying higher-order temporal
structure. Interpretation discipline: if identification accuracy survives
on AAFT surrogates, the fingerprint is attributable to marginal/linear
properties and the grammar claim must be weakened accordingly.
"""

import numpy as np


def aaft(x, rng):
    """One AAFT surrogate of 1-D series x."""
    n = len(x)
    # 1) Gaussianize: rank-remap x onto sorted Gaussian noise
    gauss = np.sort(rng.standard_normal(n))
    ranks = np.argsort(np.argsort(x))
    y = gauss[ranks]
    # 2) Phase-randomize the Gaussianized series
    fy = np.fft.rfft(y)
    phases = rng.uniform(0, 2 * np.pi, len(fy))
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    fy_s = np.abs(fy) * np.exp(1j * phases)
    y_s = np.fft.irfft(fy_s, n)
    # 3) Rescale back to the original amplitude distribution
    out = np.sort(x)[np.argsort(np.argsort(y_s))]
    return out
