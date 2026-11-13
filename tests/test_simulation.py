"""
Tests for the VG-CIR simulation, using the book parameters and checking
against formulas and prices from Schoutens (2003).
Run from the repo root: python -m pytest
"""

import numpy as np
import pytest

from vg_cir.characteristic import phi_log_price
from vg_cir.params import BOOK_MARKET, BOOK_VGCIR
from vg_cir.simulation import simulate_vgcir

P = BOOK_VGCIR
MARKET = BOOK_MARKET
T = 1.0
N_STEPS = 250
N_PATHS = 50_000


@pytest.fixture(scope="module")
def sim():
    """One simulation shared by all tests. Same setup as Table 9.1 (p. 131) but more paths. Book correction, to compare with the book's formulas"""
    return simulate_vgcir(P, MARKET, T, N_STEPS, N_PATHS, seed=42, schoutens_correction=True)


def assert_close(sample, theory, n_se=3):
    """The sample mean should be within n_se standard errors of the theoretical value."""
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    assert abs(sample.mean() - theory) < n_se * se


def test_book_parameters():
    """We use the VG-CIR parameters of Table 7.3 (p. 96) and the market data of the book"""
    assert (P.C, P.G, P.M) == (11.9896, 25.8523, 35.5344)
    assert (P.kappa, P.eta, P.lam, P.y0) == (0.6020, 1.5560, 1.9992, 1.0)
    assert (MARKET.S0, MARKET.r, MARKET.q) == (1124.47, 0.019, 0.012)


def test_cir_mean(sim):
    """E[y_T | y0], Sec 7.2.1"""
    theory = P.y0 * np.exp(-P.kappa * T) + P.eta * (1 - np.exp(-P.kappa * T))
    assert_close(sim.y[:, -1], theory)


def test_cir_variance(sim):
    """var[y_T | y0], Sec 7.2.1"""
    e = np.exp(-P.kappa * T)
    theory = (P.y0 * P.lam**2 / P.kappa * (e - e**2)
              + P.eta * P.lam**2 / (2 * P.kappa) * (1 - e) ** 2)
    yT = sim.y[:, -1]
    assert_close((yT - yT.mean()) ** 2, theory)


def test_integrated_cir_mean(sim):
    """E[Y_T | y0], Sec 7.2.1"""
    theory = P.eta * T + (P.y0 - P.eta) * (1 - np.exp(-P.kappa * T)) / P.kappa
    assert_close(sim.Y[:, -1], theory)


def test_vg_given_business_time(sim):
    """
    Given Y_T, X is VG with mean C(1/M - 1/G) Y_T and variance C(1/M^2 + 1/G^2) Y_T
    After standardising the mean is 0 and varince is 1. 
    """
    YT = sim.Y[:, -1]
    mean = P.C * (1 / P.M - 1 / P.G) * YT
    var = P.C * (1 / P.M**2 + 1 / P.G**2) * YT
    z = (sim.X[:, -1] - mean) / np.sqrt(var)
    assert_close(z, 0.0)
    assert_close(z**2, 1.0)


def test_martingale(sim):
    """The discounted stock price has mean S0 (martingale correction, Sec 7.3)."""
    discounted = np.exp(-(MARKET.r - MARKET.q) * T) * sim.S[:, -1]
    assert_close(discounted, MARKET.S0)


def test_characteristic_function(sim):
    """The simulated E[exp(iu log S_T)] matches the formula, eq. (7.4) Sec 7.3 p. 92."""
    log_ST = np.log(sim.S[:, -1])
    for u in [1.0, 5.0, 10.0]:
        simulated = np.exp(1j * u * log_ST).mean()
        assert abs(simulated - phi_log_price(u, T, P, MARKET)) < 4 / np.sqrt(N_PATHS)