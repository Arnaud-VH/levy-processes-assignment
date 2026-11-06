"""
Characteristic functions for the VG-CIR model. 
Used in the simulation of the paths and in the analytical pricer. 
Equations come from the Schoutens (2003) book.
"""

import numpy as np

def psi_vg(u, C, G, M):
    """
    Characteristic exponent of the VG process. 
    Schoutens (2003) Sec 5.3.7 using CGM parameterisation. 
    """
    u = np.asarray(u, dtype = complex)
    return C * np.log(G * M / (G * M + (M - G) * 1j * u + u**2))

def phi_Y(u, t, kappa, eta, lam, y0):
    """
    Characteristic function of the integrated CIR process. 
    Schoutens Sec 7.2.1
    """
    u = np.asarray(u, dtype = complex)
    t = np.asarray(t, dtype = float)
    gamma = np.sqrt(kappa**2 - 2 * lam**2 * 1j * u)
    half = gamma * t / 2

    #We need to be careful with coth(0) diving by 0. We take the warning and then replace at the end. 
    with np.errstate(divide = "ignore", invalid="ignore"):
        numerator = (np.exp(kappa**2 * eta * t / lam**2)) * np.exp(2 * y0 * 1j * u / (kappa + gamma / np.tanh(half)))
        denominator = (np.cosh(half) + kappa * np.sinh(half) / gamma) ** (2 * kappa * eta / lam**2)
        phi = numerator / denominator 

    #We know at t=0 the value is 1, so we replace. 
    return np.where(t == 0, 1.0 + 0j, phi)

def martingale_correction(params, t):
    """
    Doing the martingale corrrection makes the discounted stock price a martingale. 
    Needs M > 1, otherwise psi_X(-i) is not defined. 
    Schoutens Sec 7.3.
    """
    p = params
    psi = psi_vg(-1j, p.C, p.G, p.M)
    return phi_Y(-1j * psi, t, p.kappa, p.eta, p.lam, p.y0).real

def phi_log_price(u, T, params, market):
    """
    Characteristic function of log(S_T) under VG-CIR. Schoutens section 7.3 equation 7.4.
    Used for the analytical call pricer. 
    """
    p = params
    u = np.asarray(u, dtype = complex)
    drift = np.exp(1j * u * ((market.r - market.q) * T + np.log(market.S0)))
    numerator = phi_Y(-1j * psi_vg(u, p.C, p.G, p.M), T, p.kappa, p.eta, p.lam, p.y0)
    return drift * numerator / martingale_correction(p, T) ** (1j * u)