"""
Part 2 of the project, used in part 4.
Call option prices from the analytical formula (Schoutens, p. 20), used in the estimation.

This is the same formula as call_price in scripts/02_estimate.py (Tiago), but written with our
VGCIRParams and MarketParams so we can import it without running the whole calibration.
"""

import numpy as np
from scipy.integrate import simpson

from vg_cir.characteristic import phi_Y, psi_vg

#Integration grid for u, the one used in the calibration in the report (Simpson rule, 3001 points over [1e-6, 300])
U = np.linspace(1e-6, 300, 3001)


def phi_log_price(u, T, params, market):
    """
    Characteristic function of log(S_T) with the correction used in the calibration:
    """
    p = params
    u = np.asarray(u, dtype=complex)
    exponent = psi_vg(u, p.C, p.G, p.M) - 1j * u * psi_vg(-1j, p.C, p.G, p.M)
    drift = np.exp(1j * u * (np.log(market.S0) + (market.r - market.q) * T))
    return drift * phi_Y(-1j * exponent, T, p.kappa, p.eta, p.lam, p.y0)


def call_price(K, T, params, market):
    """
    European call price with formulas (2.3)-(2.5) in Schoutens p. 20:
    C = S0 exp(-qT) P1 - K exp(-rT) P2, where P1 and P2 are integrals of the characteristic function.
    """
    def integrand1(u):
        return np.real(np.exp(-1j * u * np.log(K)) * phi_log_price(u - 1j, T, params, market)
                       / (1j * u * phi_log_price(-1j, T, params, market)))

    def integrand2(u):
        return np.real(np.exp(-1j * u * np.log(K)) * phi_log_price(u, T, params, market) / (1j * u))

    P1 = 0.5 + simpson(integrand1(U), x=U) / np.pi
    P2 = 0.5 + simpson(integrand2(U), x=U) / np.pi
    return market.S0 * np.exp(-market.q * T) * P1 - K * np.exp(-market.r * T) * P2
