"""
Tests for the Monte Carlo pricer, using the book parameters.
test_mc_call_matches_analytical uses our estimated parameters and compares with the analytical pricer.
Run from the repo root: python -m pytest
"""

import numpy as np
import pytest

from vg_cir.params import BOOK_MARKET, BOOK_VGCIR, DATASET_MARKET, ESTIMATED_VGCIR
from vg_cir.pricing.analytical import call_price
from vg_cir.pricing.montecarlo import EuropeanCall, LookbackCall, UpAndOutCall
from vg_cir.simulation import simulate_vgcir

MARKET = BOOK_MARKET
T = 1.0
K = MARKET.S0


@pytest.fixture(scope="module")
def S():
    """Stock paths shared by all tests."""
    return simulate_vgcir(BOOK_VGCIR, MARKET, T, n_steps=250, n_paths=20_000, seed=42).S


def test_payoffs_by_hand():
    """Three tiny paths where we know the payoff: in the money, out of the money, knocked out."""
    #path 1 ends at 110 and never touches 130, path 2 ends below the strike, path 3 touches the barrier and comes back down
    S = np.array([[100.0, 105.0, 110.0],
                  [100.0,  95.0,  90.0],
                  [100.0, 135.0, 110.0]])
    call = EuropeanCall(strike=100.0, maturity=T)
    uobc = UpAndOutCall(strike=100.0, maturity=T, barrier=130.0)
    lookback = LookbackCall(maturity=T)
    assert call.payoff(S).tolist() == [10.0, 0.0, 10.0]
    assert uobc.payoff(S).tolist() == [10.0, 0.0, 0.0]
    #lookback pays S_T - min: 110 - 100, 90 - 90, 110 - 100
    assert lookback.payoff(S).tolist() == [10.0, 0.0, 10.0]


def test_barrier_option_is_cheaper(S):
    """Knocking out can only remove payoff, so UOBC <= plain call."""
    call = EuropeanCall(K, T).price(S, MARKET)
    uobc = UpAndOutCall(K, T, barrier=1.2 * K).price(S, MARKET)
    assert uobc.price <= call.price


def test_unreachable_barrier_is_plain_call(S):
    """With a barrier no path reaches, the UOBC is exactly the plain call."""
    call = EuropeanCall(K, T).price(S, MARKET)
    uobc = UpAndOutCall(K, T, barrier=100 * K).price(S, MARKET)
    assert uobc.price == pytest.approx(call.price)


def test_barrier_must_be_above_strike():
    with pytest.raises(ValueError):
        UpAndOutCall(K, T, barrier=0.5 * K)


def test_lookback_matches_book():
    """
    Lookback call with the book's parameters and setup (T = 1, 250 steps, book correction)
    against Table 9.1 (p. 131): VG-CIR LC = 135.27 with standard error 0.4942.
    Both prices are Monte Carlo estimates, so we allow 3 times the combined standard error.
    """
    S_book = simulate_vgcir(BOOK_VGCIR, MARKET, T, n_steps=250, n_paths=20_000, seed=42, schoutens_correction=True).S
    mc = LookbackCall(T).price(S_book, MARKET)
    combined_error = np.sqrt(mc.std_error**2 + 0.4942**2)
    assert abs(mc.price - 135.27) < 3 * combined_error


def test_mc_call_matches_analytical():
    """
    With our estimated parameters, the MC price of a plain call should be within 3 standard errors
    of the analytical price (the formula used in the calibration). Same setup as the UOBC in the assignment: T = 9 months, 270 steps.
    """
    S = simulate_vgcir(ESTIMATED_VGCIR, DATASET_MARKET, 0.75, n_steps=270, n_paths=20_000, seed=42).S
    for k in [0.9, 1.0, 1.1]:
        strike = k * DATASET_MARKET.S0
        mc = EuropeanCall(strike, 0.75).price(S, DATASET_MARKET)
        exact = call_price(strike, 0.75, ESTIMATED_VGCIR, DATASET_MARKET)
        assert abs(mc.price - exact) < 3 * mc.std_error
