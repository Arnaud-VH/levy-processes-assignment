"""
Part 2 of the project, used in part 4.
Black-Scholes formulas, to compare against the VG-CIR model.
The up-and-out call has a closed form under Black-Scholes (Schoutens section 9.1.2, p. 121-122),
so we can check our Monte Carlo barrier code against an exact price, and compare the two models.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

#Barrier shift for a barrier that is only checked at discrete dates, Schoutens p. 121-122
BGK_BETA = 0.582


def bs_call(K, T, sigma, market):
    """Black-Scholes price of a European call with dividend yield q."""
    v = sigma * np.sqrt(T)
    d1 = (np.log(market.S0 / K) + (market.r - market.q + 0.5 * sigma**2) * T) / v
    d2 = d1 - v
    return market.S0 * np.exp(-market.q * T) * ndtr(d1) - K * np.exp(-market.r * T) * ndtr(d2)

def implied_vol(price, K, T, market):
    """The sigma for which the Black-Scholes call price equals price (root finding with brentq)."""
    return brentq(lambda sigma: bs_call(K, T, sigma, market) - price, 1e-4, 3.0)

def bs_up_and_out_call(K, T, H, sigma, market):
    """
    Black-Scholes price of the up-and-out barrier call, with the barrier checked continuously.
    Schoutens p. 121-122: if H <= K the option can never pay, so UOBC = 0.
    If H > K, UOBC = C - UIBC, with the formula for the up-and-in call UIBC.
    """
    S0, r, q = market.S0, market.r, market.q
    if H <= K or H <= S0:
        return 0.0
    v = sigma * np.sqrt(T)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    y = np.log(H**2 / (S0 * K)) / v + lam * v
    x1 = np.log(S0 / H) / v + lam * v
    y1 = np.log(H / S0) / v + lam * v

    uibc = (S0 * ndtr(x1) * np.exp(-q * T) - K * np.exp(-r * T) * ndtr(x1 - v) 
            - S0 * np.exp(-q * T) * (H / S0)**(2 * lam) * (ndtr(-y) - ndtr(-y1)) 
            + K * np.exp(-r * T) * (H / S0)**(2 * lam - 2) * (ndtr(-y + v) - ndtr(-y1 + v)))
    return bs_call(K, T, sigma, market) - uibc


def discrete_barrier(H, sigma, dt):
    """
    Shifts the barrier so that the continuous formula gives the price of a barrier that is checked every dt years.
    For an up barrier we shift it up, because between two checks a path can go above H and come back down without us seeing it.
    """
    return H * np.exp(BGK_BETA * sigma * np.sqrt(dt))


def simulate_bs(sigma, market, T, n_steps, n_paths, seed=None):
    """
    Black-Scholes stock price paths (geometric Brownian motion), exact scheme, Schoutens section 8.1.
    Returns S with shape (n_paths, n_steps + 1), same layout as simulate_vgcir.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = (market.r - market.q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = market.S0
    S[:, 1:] = market.S0 * np.exp(np.cumsum(log_returns, axis=1))
    return S
