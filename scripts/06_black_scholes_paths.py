"""
Part 3 of project, benchmark for the sensitivity analysis (Samuel).
Simulates Black-Scholes paths with the volatility fitted in Part 2, and plots the paths,
the distribution of the terminal log-returns and its moments, to compare with VG-CIR (03_simulate_paths.py).
Run from the repo root: python scripts/06_black_scholes_paths.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vg_cir.distribution import calculate_moments, plot_empirical_pdf, plot_moments_table
from vg_cir.params import BS_SIGMA, DATASET_MARKET
from vg_cir.pricing.black_scholes import simulate_bs

#Folder of the repo (one above scripts/), so the outputs go to the same results/ wherever the script is run from
FIG_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

MARKET = DATASET_MARKET
SIGMA = BS_SIGMA

T = 1.0
N_STEPS = 250
N_PATHS = 100_000  #for the empirical distribution
SEED = 2027
SHOWN = 15  #paths shown in the plot


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    #Same exact scheme as used in Part 4 (pricing/black_scholes.py)
    S = simulate_bs(SIGMA, MARKET, T, N_STEPS, N_PATHS, seed=SEED)
    t = np.linspace(0.0, T, N_STEPS + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, S[:SHOWN].T, lw=0.8)
    ax.plot(t, MARKET.S0 * np.exp((MARKET.r - MARKET.q) * t), "k--", lw=1.5, label="$S_0 e^{(r-q)t}$ (mean)")
    ax.set_title(f"$S_t$ (stock price) path under Black-Scholes (volatility = {SIGMA:.4f})")
    ax.set_xlabel("t (years)")
    ax.set_ylabel("$S_t$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bs_paths.png", dpi=150)

    #Empirical distribution of the terminal log-returns
    fig_dist = plot_empirical_pdf(S, MARKET.S0, n_bins=60, title=f"Black-Scholes Empirical PDF (volatility = {SIGMA:.4f})")
    fig_dist.savefig(FIG_DIR / "bs_distribution.png", dpi=150)

    moments = calculate_moments(S, MARKET.S0)
    plot_moments_table({"Base": moments}, FIG_DIR / "bs_moments_table.png")
    print("Moments of log(S_T/S0) under Black-Scholes:", {k: round(float(v), 4) for k, v in moments.items()})

    plt.show()
