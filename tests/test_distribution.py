"""
Tests for distribution.py: the bias corrected moments should match scipy's versions.
"""

import numpy as np
from scipy import stats

from vg_cir.distribution import calculate_moments


def test_moments_match_scipy():
    rng = np.random.default_rng(0)
    #Paths with a skewed, fat tailed terminal value, only the last column matters
    S = np.ones((5000, 2))
    S[:, -1] = np.exp(rng.standard_t(df=5, size=5000) * 0.1 - 0.05 * rng.exponential(size=5000))
    log_r = np.log(S[:, -1])

    m = calculate_moments(S, 1.0)
    assert np.isclose(m["Mean"], log_r.mean())
    assert np.isclose(m["Variance"], log_r.var(ddof=1))
    assert np.isclose(m["Skewness"], stats.skew(log_r, bias=False))
    #scipy gives the excess kurtosis, ours is the plain kurtosis (3 for a normal)
    assert np.isclose(m["Kurtosis"], stats.kurtosis(log_r, bias=False) + 3)
