"""
Parts 3 and 4 of the project.
Implementation of the simulation for the VG-CIR model.
"""

from dataclasses import dataclass

import numpy as np
from vg_cir.characteristic import martingale_correction, psi_vg

def simulate_cir(kappa, eta, lam, y0, T, n_steps, n_paths, rng):
    """
    Simulates the CIR rate of time change y with an Euler scheme. 
    From Section 8.4.8 in Schoutens (2003)
    Implemented max(y,0) inside the drift and the square root to ensure no errors. 
    This is because the Feller condition 2*kappa*eta > lam^2 fails for the book parameters
    so y can reach 0 and an euler step can go negative. 

    We take rng as a parameter so the seed can be controlled in the place the function is called. 
    Returns:
        y with shape (n_paths, n_steps + 1). Column 0 is y0
    """

    dt = T / n_steps
    #v_n from the book
    v = rng.standard_normal((n_paths, n_steps))

    y = np.empty((n_paths, n_steps + 1))
    y[:, 0] = y0
    y_raw = np.full(n_paths, float(y0))

    for n in range(1, n_steps + 1):
        y_pos = np.maximum(y_raw, 0.0)
        y_raw = y_raw + kappa * (eta - y_pos) * dt + lam * np.sqrt(y_pos) * np.sqrt(dt) * v[:, n - 1]
        y[:, n] = np.maximum(y_raw, 0.0)

    return y

def integrate_cir(y, dt):
    """
    Business time, integral of y_t process, using  left Riemann sum. 
    Schoutens (2003) section 7.2.1.
    Returns Y with the same shape as y, and Y_0 = 0. 
    """
    Y = np.zeros_like(y)
    Y[:, 1:] = np.cumsum(y[:, :-1] * dt, axis = 1)
    return Y


def simulate_vg_increments(C, G, M, dY, rng_up, rng_down):
    """
    Increments of the time changed VG process X_{Y_t} over each time step.
    We take VG as the difference of two independent Gamma processes, Schoutens (2003) Section 8.4.2.
    NumPy's gamma uses scale = 1 / rate, so scale = 1/M (up jumps) and 1/G (down jumps).
    """
    up = rng_up.gamma(shape=C * dY, scale=1.0 / M)
    down = rng_down.gamma(shape=C * dY, scale=1.0 / G)
    return up - down


@dataclass
class SimulationResult:
    """
    Output of simulate_vgcir. t has shape (n_steps + 1,), the rest (n_paths, n_steps + 1).
    """
    t: np.ndarray
    S: np.ndarray
    y: np.ndarray    
    Y: np.ndarray    
    X: np.ndarray


def simulate_vgcir(params, market, T, n_steps, n_paths, seed=None, schoutens_correction=False):
    """
    Simulates VG-CIR stock price paths, following Schoutens (2003) section 9.3.2, all 5 steps.
    By default we use the same martingale correction as the calibration in 02_estimate.py: exp(X_{Y_t} - psi(-i) Y_t).
    With schoutens_correction=True we divide by E[exp(X_{Y_t})] like the book (section 7.3),
    this is only used to check our simulation against the book's numbers.
    """
    p = params
    rng_cir, rng_up, rng_down = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]

    t = np.linspace(0.0, T, n_steps + 1)
    dt = T / n_steps

    #(i) rate of time change
    y = simulate_cir(p.kappa, p.eta, p.lam, p.y0, T, n_steps, n_paths, rng_cir)
    #(ii) business time
    Y = integrate_cir(y, dt)

    #(iii)-(iv) VG increments drawn directly over each business-time step, X_{Y_0} = 0
    dX = simulate_vg_increments(p.C, p.G, p.M, np.diff(Y, axis=1), rng_up, rng_down)
    X = np.zeros_like(Y)
    X[:, 1:] = np.cumsum(dX, axis=1)

    #(v) stock price with the martingale correction
    if schoutens_correction:
        #book version (section 7.3), deterministic correction
        correction = martingale_correction(p, t)
        S = market.S0 * np.exp((market.r - market.q) * t) * np.exp(X) / correction
    else:
        #calibration version, the correction runs on business time so exp(X - psi(-i) Y) is a martingale along the whole path
        psi = psi_vg(-1j, p.C, p.G, p.M).real
        S = market.S0 * np.exp((market.r - market.q) * t) * np.exp(X - psi * Y)

    return SimulationResult(t=t, S=S, y=y, Y=Y, X=X)