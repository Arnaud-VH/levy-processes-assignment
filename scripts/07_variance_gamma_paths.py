"""
Part 3 of project, benchmark for the sensitivity analysis (Samuel).
Simulates the Variance Gamma model without the stochastic clock (deterministic time), with the parameters
estimated for the VG benchmark in Part 2, and plots the paths, the distribution of the terminal log-returns and its moments.
Run from the repo root: python scripts/07_variance_gamma_paths.py
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vg_cir.distribution import calculate_moments, plot_empirical_pdf, plot_moments_table
from vg_cir.params import DATASET_MARKET, VG_A, VG_MU, VG_SIGMA
from vg_cir.simulation import simulate_vg_increments  #same VG increments as in the VG-CIR simulation

#Folder of the repo (one above scripts/), so the outputs go to the same results/ wherever the script is run from
FIG_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

MARKET = DATASET_MARKET

T = 1.0
N_STEPS = 250
N_PATHS = 100_000  #for the empirical distribution
SEED = 2027
SHOWN = 15  #paths shown in the plot


@dataclass
class VGSimulationResult:
    t: np.ndarray
    X: np.ndarray
    S: np.ndarray


def vg_to_cgm(mu, sigma, a):
    """
    (mu, sigma, a) of the report to (C, G, M), Schoutens (2003) Section 5.3.7, with nu = 1/a.
    Same as the conversion in Section 5.2.3 of the report.
    """
    C = a
    nu = 1 / C
    term = np.sqrt(0.25 * mu**2 * nu**2 + 0.5 * sigma**2 * nu)
    G = 1 / (term - 0.5 * mu * nu)
    M = 1 / (term + 0.5 * mu * nu)
    return C, G, M


def simulate_vg(market, T, n_steps, n_paths, C, G, M, seed=None):
    """
    Variance Gamma stock price paths, VG increments as a difference of two Gamma variables over each time step.
    """
    rng_up, rng_down = np.random.default_rng(seed).spawn(2)

    t = np.linspace(0.0, T, n_steps + 1)
    dt = T / n_steps

    #The time change is deterministic, so every step has length dt for every path
    dY = np.full((n_paths, n_steps), dt)
    dX = simulate_vg_increments(C, G, M, dY, rng_up, rng_down)
    X = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dX, axis=1)], axis=1)

    #Mean-correcting martingale measure, Schoutens (2003) Table 6.2, p. 80
    omega = C * np.log(((M - 1) * (G + 1)) / (M * G))
    S = market.S0 * np.exp((market.r - market.q + omega) * t[None, :] + X)

    return VGSimulationResult(t=t, X=X, S=S)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    C, G, M = vg_to_cgm(VG_MU, VG_SIGMA, VG_A)
    sim = simulate_vg(MARKET, T, N_STEPS, N_PATHS, C, G, M, SEED)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sim.t, sim.S[:SHOWN].T, lw=0.8)
    ax.plot(sim.t, MARKET.S0 * np.exp((MARKET.r - MARKET.q) * sim.t), "k--", lw=1.5, label="$S_0 e^{(r-q)t}$ (mean)")
    ax.set_title(f"$S_t$ path under Variance Gamma (C={C:.4f}, G={G:.4f}, M={M:.4f})")
    ax.set_xlabel("t (years)")
    ax.set_ylabel("$S_t$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vg_paths.png", dpi=150)

    #Empirical distribution of the terminal log-returns
    fig_dist = plot_empirical_pdf(sim.S, MARKET.S0, n_bins=60, title=f"Variance Gamma Empirical PDF (C={C:.2f}, G={G:.2f}, M={M:.2f})")
    fig_dist.savefig(FIG_DIR / "vg_distribution.png", dpi=150)

    moments = calculate_moments(sim.S, MARKET.S0)
    plot_moments_table({"Base": moments}, FIG_DIR / "vg_moments_table.png")
    print("Moments of log(S_T/S0) under Variance Gamma:", {k: round(float(v), 4) for k, v in moments.items()})

    plt.show()
