"""
Part 3 of project.
Simulates paths of the VG-CIR model with simulation.py and saves plots.
Run from the repo root: python scripts/03_simulate_paths.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vg_cir.params import ESTIMATED_VGCIR, DATASET_MARKET
from vg_cir.simulation import simulate_vgcir

#Folder of the repo (one above scripts/), so the outputs go to the same results/ wherever the script is run from
FIG_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

#Estimated parameters from Part 2
PARAMS = ESTIMATED_VGCIR
MARKET = DATASET_MARKET

T = 1.0
N_STEPS = 250
SEED = 42

def plot_components(sim):
    """One path of every layer of the model, like Fig. 9.1 in Schoutens (p. 129)."""
    fig, axes = plt.subplots(5, 1, figsize=(8, 10))
    axes[0].plot(sim.t, sim.y[0])
    axes[0].set_title("$y_t$ (rate of time change)")

    axes[1].plot(sim.t, sim.Y[0])
    axes[1].set_title("$Y_t$ (business time)")

    axes[2].plot(sim.Y[0], sim.X[0])
    axes[2].set_title("$X_t$ on $[0, Y_T]$ (VG in business time)")
    axes[2].set_xlabel("business time $Y$")

    axes[3].plot(sim.t, sim.X[0])
    axes[3].set_title("$X_{Y_t}$ (time-changed VG)")


    axes[4].plot(sim.t, sim.S[0])
    axes[4].set_title("$S_t$ (stock price)")
    axes[4].set_xlabel("t (years)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vgcir_components.png", dpi=150)


def plot_many_paths(sim, market, shown=20):
    """The first shown paths of y, X_{Y_t} and S."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(sim.t, sim.y[:shown].T, lw=0.8)
    axes[0].set_title("$y_t$ (rate of time change)")

    axes[1].plot(sim.t, sim.X[:shown].T, lw=0.8)
    axes[1].set_title("$X_{Y_t}$ (time-changed VG)")

    axes[2].plot(sim.t, sim.S[:shown].T, lw=0.8)
    axes[2].plot(sim.t, market.S0 * np.exp((market.r - market.q) * sim.t), "k--", lw=1.5, label="$S_0 e^{(r-q)t}$ (mean of $S_t$)")
    axes[2].set_title("$S_t$ (stock price)")
    axes[2].set_xlabel("t (years)")
    axes[2].legend()

    fig.suptitle(f"Showing {shown}/{sim.S.shape[0]} simulated VG-CIR paths")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vgcir_paths.png", dpi=150)


if __name__ == "__main__":
    #Create the directory to store the results 
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sim = simulate_vgcir(PARAMS, MARKET, T, N_STEPS, n_paths=100, seed=SEED)
    plot_components(sim)
    plot_many_paths(sim, MARKET)

    plt.show()
