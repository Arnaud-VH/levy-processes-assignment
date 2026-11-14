"""
Part 3 of project.
Simulates paths of the VG-CIR model with simulation.py and saves plots.
Also the distribution of the terminal log-returns, and the sensitivity analysis of the parameters (Samuel):
every parameter is changed on its own (SENSITIVITY_CASES in params.py) and the paths, the distribution and the moments
are compared with the estimated parameters.
Run from the repo root: python scripts/03_simulate_paths.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vg_cir.distribution import calculate_moments, log_returns, plot_empirical_pdf, plot_moments_table
from vg_cir.params import ESTIMATED_VGCIR, DATASET_MARKET, SENSITIVITY_CASES
from vg_cir.simulation import simulate_vgcir

#Folder of the repo (one above scripts/), so the outputs go to the same results/ wherever the script is run from
FIG_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
SENS_DIR = FIG_DIR / "sensitivity"

#Estimated parameters from Part 2
PARAMS = ESTIMATED_VGCIR
MARKET = DATASET_MARKET

T = 1.0
N_STEPS = 250
SEED = 42

#Distribution of the returns and sensitivity analysis: many paths for the histograms, fewer paths shown in the plots
N_PATHS_DIST = 100_000
SENS_SEED = 2027
SENS_SHOWN = 15
#The sensitivity analysis simulates 10 x 100 000 paths, so it is off by default. Set to True to run it.
RUN_SENSITIVITY = False


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


def plot_many_paths(sim, market, shown=20, path=None, title_suffix=""):
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

    fig.suptitle(f"Showing {shown}/{sim.S.shape[0]} simulated VG-CIR paths {title_suffix}")
    fig.tight_layout()
    fig.savefig(path if path is not None else FIG_DIR / "vgcir_paths.png", dpi=150)
    return fig


def params_title(p):
    return (f"$C$={p.C:.4f}, $G$={p.G:.4f}, $M$={p.M:.4f}, "
            f"$\\kappa$={p.kappa:.4f}, $\\eta$={p.eta:.4f}, $\\lambda$={p.lam:.4f}")


def file_tag(name):
    """Scenario name to a file name, for example 'C +20%' -> 'Cplus20' and 'G > M' -> 'GgtM'."""
    name = name.replace("+", "plus").replace("-", "minus").replace("=", "eq").replace(">", "gt")
    return re.sub(r"[^A-Za-z0-9]", "", name)


def run_scenario(name, params, base_mu, base_sigma):
    """
    Paths, distribution of the returns and moments for one set of parameters.
    The normal curve uses the mean and std of the base case, so every histogram is compared with the same curve.
    """
    sim = simulate_vgcir(params, MARKET, T, N_STEPS, n_paths=N_PATHS_DIST, seed=SENS_SEED)
    tag = file_tag(name)
    fig = plot_many_paths(sim, MARKET, shown=SENS_SHOWN, path=SENS_DIR / f"vgcir_paths_{tag}.png", title_suffix=f"({name})")
    plt.close(fig)
    fig = plot_empirical_pdf(sim.S, MARKET.S0, n_bins=60, title=f"VG-CIR Empirical PDF ({params_title(params)})",
                             fixed_mu=base_mu, fixed_sigma=base_sigma)
    fig.savefig(SENS_DIR / f"vgcir_distribution_{tag}.png", dpi=150)
    plt.close(fig)
    return calculate_moments(sim.S, MARKET.S0)


if __name__ == "__main__":
    #Create the directory to store the results
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sim = simulate_vgcir(PARAMS, MARKET, T, N_STEPS, n_paths=100, seed=SEED)
    plot_components(sim)
    plot_many_paths(sim, MARKET)

    #Distribution of the terminal log-returns with the estimated parameters
    sim_base = simulate_vgcir(PARAMS, MARKET, T, N_STEPS, n_paths=N_PATHS_DIST, seed=SENS_SEED)
    base_r = log_returns(sim_base.S, MARKET.S0)
    base_mu, base_sigma = base_r.mean(), base_r.std()
    base_moments = calculate_moments(sim_base.S, MARKET.S0)
    fig = plot_empirical_pdf(sim_base.S, MARKET.S0, n_bins=60, title=f"VG-CIR Empirical PDF ({params_title(PARAMS)})")
    fig.savefig(FIG_DIR / "vgcir_distribution.png", dpi=150)
    plot_moments_table({"Base": base_moments}, FIG_DIR / "vgcir_moments_base.png")
    print("Moments of log(S_T/S0) with the estimated parameters:", {k: round(float(v), 4) for k, v in base_moments.items()})

    #Sensitivity analysis: one parameter changed at a time, one moments table per parameter
    if RUN_SENSITIVITY:
        SENS_DIR.mkdir(parents=True, exist_ok=True)
        fig = plot_many_paths(sim_base, MARKET, shown=SENS_SHOWN, path=SENS_DIR / "vgcir_paths_base.png", title_suffix="(Base)")
        plt.close(fig)
        del sim_base

        for parameter, cases in SENSITIVITY_CASES.items():
            results = {"Base": base_moments}
            for name, params in cases.items():
                results[name] = run_scenario(name, params, base_mu, base_sigma)
                print(name, {k: round(float(v), 4) for k, v in results[name].items()})
            fig = plot_moments_table(results, SENS_DIR / f"vgcir_moments_{parameter}.png")
            plt.close(fig)

    plt.show()
