"""
Part 4 of project.
Prices the up and out barrier call by Monte Carlo (using montecarlo.py), using paths from simulation.py.
Then studies the effect of the barrier H and the time to maturity T on the price.
Also compares Monte Carlo with the analytical pricer (analytical.py) on a plain call to check that montecarlo.py pricer gives the right prices.
Run from the repo root: python scripts/04_price_barrier.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vg_cir.params import ESTIMATED_VGCIR, DATASET_MARKET
from vg_cir.pricing.analytical import call_price as analytical_call
from vg_cir.pricing.montecarlo import EuropeanCall, LookbackCall, UpAndOutCall
from vg_cir.simulation import simulate_vgcir

#Folder of the repo (one above scripts/), so the outputs go to the same results/ wherever the script is run from
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"

#Estimated parameters from Part 2
PARAMS = ESTIMATED_VGCIR
MARKET = DATASET_MARKET

#Contract from the assignment: 9 month up-and-out call, K = S0, H = 1.25 S0
T = 0.75
STRIKE = MARKET.S0
BARRIER = 1.25 * MARKET.S0

#Assignment asks for 2000 paths and 270 steps. With 2000 paths the standard error is about 3% of the price,
#so we also report a run with more paths.
N_STEPS = 270
N_PATHS = 2_000
SEED = 42

#Sensitivity study of H and T. We use more paths than the assignment so the curves are smooth
#and the same paths for every point so the differences come from H and T and not from noise.
N_PATHS_SWEEP = 20_000
#DT = 1/360, we keep it the same for every maturity, so all maturities below have to be multiples of DT
DT = T / N_STEPS
H_RATIOS = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.60, 1.80]
MATURITIES = [0.10, 0.25, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]


def print_price(name, result):
    low = result.price - 1.96 * result.std_error
    high = result.price + 1.96 * result.std_error
    print(f"{name:<22} {result.price:10.4f}   std error {result.std_error:.4f}   95% CI [{low:.4f}, {high:.4f}]")


def plot_barrier_paths(sim, option, shown=20):
    """A few paths with the barrier, knocked-out paths in red."""
    fig, ax = plt.subplots(figsize=(8, 5))
    out = option.knocked_out(sim.S[:shown])
    ax.plot(sim.t, sim.S[:shown][~out].T, color="tab:blue", lw=0.8)
    ax.plot(sim.t, sim.S[:shown][out].T, color="tab:red", lw=0.8)
    ax.axhline(option.barrier, color="k", ls="--", label=f"barrier H = {option.barrier:.0f}")
    ax.axhline(option.strike, color="gray", ls=":", label=f"strike K = {option.strike:.0f}")
    ax.set_title(f"{out.sum()}/{shown} shown paths knocked out")
    ax.set_xlabel("t (years)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "uobc_paths.png", dpi=150)


def cut_paths(S, maturity):
    """
    Cuts the paths at time maturity. We simulate once up to the longest maturity with step DT
    so the path up to T is just the first T/DT steps and we don't have to simulate again for every T.
    """
    n = round(maturity / DT)
    return S[:, :n + 1]

#Here we start doing the analysis of the barrier option price w.r.t the barrier and the maturity
#Also did a grid as additional material to view both. 
def barrier_sweep(S):
    """
    UOBC price for every barrier H = h * S0 in H_RATIOS, with T = 9 months.
    Only the barrier changes, the paths are the same for every H.
    """
    S_T = cut_paths(S, T)
    rows = []
    for h in H_RATIOS:
        uobc = UpAndOutCall(STRIKE, T, h * MARKET.S0)
        result = uobc.price(S_T, MARKET)
        rows.append({"H/S0": h, "UOBC": result.price, "std_error": result.std_error,
                     "knocked_out": uobc.knocked_out(S_T).mean()})
    return pd.DataFrame(rows)


def maturity_sweep(S):
    """
    UOBC and plain call price for every maturity in MATURITIES, with H = 1.25 S0.
    """
    rows = []
    for maturity in MATURITIES:
        S_cut = cut_paths(S, maturity)
        uobc = UpAndOutCall(STRIKE, maturity, BARRIER)
        result = uobc.price(S_cut, MARKET)
        call = EuropeanCall(STRIKE, maturity).price(S_cut, MARKET)
        rows.append({"T": maturity, "UOBC": result.price, "std_error": result.std_error,
                     "call": call.price, "call_std_error": call.std_error,
                     "call_analytical": analytical_call(STRIKE, maturity, PARAMS, MARKET),
                     "knocked_out": uobc.knocked_out(S_cut).mean()})
    return pd.DataFrame(rows)


def grid_sweep(S):
    """
    UOBC price for every pair (T, H) in MATURITIES x H_RATIOS, on the same paths.
    Returns two tables with T as rows and H/S0 as columns: the prices and the standard errors.
    """
    prices = pd.DataFrame(index=MATURITIES, columns=H_RATIOS, dtype=float)
    errors = pd.DataFrame(index=MATURITIES, columns=H_RATIOS, dtype=float)
    for maturity in MATURITIES:
        S_cut = cut_paths(S, maturity)
        for h in H_RATIOS:
            result = UpAndOutCall(STRIKE, maturity, h * MARKET.S0).price(S_cut, MARKET)
            prices.loc[maturity, h] = result.price
            errors.loc[maturity, h] = result.std_error
    prices.index.name = errors.index.name = "T"
    prices.columns.name = errors.columns.name = "H/S0"
    return prices, errors


def plot_grid(prices):
    """Heatmap of the UOBC price over the (T, H) grid, with the price written in every cell."""
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(prices.values, cmap="viridis", aspect="auto", origin="lower")
    for i in range(len(MATURITIES)):
        for j in range(len(H_RATIOS)):
            value = prices.values[i, j]
            color = "white" if value < prices.values.max() / 2 else "black"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_xticks(range(len(H_RATIOS)), [f"{h:.2f}" for h in H_RATIOS])
    ax.set_yticks(range(len(MATURITIES)), [f"{t:.2f}" for t in MATURITIES])
    ax.set_xlabel("$H / S_0$")
    ax.set_ylabel("T (years)")
    ax.set_title("UOBC price for every barrier and maturity (K = $S_0$)")
    fig.colorbar(image, ax=ax, label="price")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "uobc_grid.png", dpi=150)


def plot_barrier_sweep(table, call_price):
    """UOBC price against H/S0, with a 95% band and the plain call as the limit for large H."""
    fig, ax = plt.subplots(figsize=(8, 5))
    band = 1.96 * table["std_error"]
    ax.plot(table["H/S0"], table["UOBC"], "o-", label="UOBC")
    ax.fill_between(table["H/S0"], table["UOBC"] - band, table["UOBC"] + band, alpha=0.3, label="95% CI")
    ax.axhline(call_price, color="k", ls="--", label="plain call")
    ax.axvline(BARRIER / MARKET.S0, color="gray", ls=":", label="H = 1.25 $S_0$ (assignment)")
    ax.set_title(f"UOBC price against the barrier (T = {T}, K = $S_0$)")
    ax.set_xlabel("$H / S_0$")
    ax.set_ylabel("price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "uobc_barrier.png", dpi=150)


def plot_maturity_sweep(table):
    """UOBC and plain call price against T."""
    fig, ax = plt.subplots(figsize=(8, 5))
    band = 1.96 * table["std_error"]
    ax.plot(table["T"], table["UOBC"], "o-", label="UOBC")
    ax.fill_between(table["T"], table["UOBC"] - band, table["UOBC"] + band, alpha=0.3, label="95% CI")
    ax.plot(table["T"], table["call"], "k--", label="plain call")
    ax.axvline(T, color="gray", ls=":", label="T = 9 months (assignment)")
    ax.set_title("UOBC price against the time to maturity (K = $S_0$, H = 1.25 $S_0$)")
    ax.set_xlabel("T (years)")
    ax.set_ylabel("price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "uobc_maturity.png", dpi=150)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sim = simulate_vgcir(PARAMS, MARKET, T, N_STEPS, N_PATHS, seed=SEED)

    call = EuropeanCall(STRIKE, T)
    uobc = UpAndOutCall(STRIKE, T, BARRIER)

    mc_call = call.price(sim.S, MARKET)
    mc_uobc = uobc.price(sim.S, MARKET)

    #Check of the pricer: the MC plain call should be close to the analytical price (within ~2 std errors)
    exact_call = analytical_call(STRIKE, T, PARAMS, MARKET)
    print_price("Plain call (MC)", mc_call)
    print(f"{'Plain call (analytical)':<22} {exact_call:10.4f}   difference = {(mc_call.price - exact_call) / mc_call.std_error:.2f} std errors")
    print_price("Up-and-out call (MC)", mc_uobc)
    print(f"Knocked out: {uobc.knocked_out(sim.S).mean():.1%} of paths")

    #Shows our OOP design by being able to easily have new options added
    #Any other option only needs a payoff, it is priced on the same paths with the same price() method
    print_price("Lookback call (MC)", LookbackCall(T).price(sim.S, MARKET))

    plot_barrier_paths(sim, uobc)

    #Effect of H and T: one simulation up to the longest maturity, used for both studies
    t_max = max(MATURITIES)
    sweep_sim = simulate_vgcir(PARAMS, MARKET, t_max, round(t_max / DT), N_PATHS_SWEEP, seed=SEED)

    barrier_table = barrier_sweep(sweep_sim.S)
    call_price = EuropeanCall(STRIKE, T).price(cut_paths(sweep_sim.S, T), MARKET).price
    print(f"\nEffect of the barrier H (T = {T}, {N_PATHS_SWEEP} paths), plain call = {call_price:.2f}")
    print(barrier_table.round(4).to_string(index=False))

    maturity_table = maturity_sweep(sweep_sim.S)
    print(f"\nEffect of the maturity T (H = 1.25 S0, {N_PATHS_SWEEP} paths)")
    print(maturity_table.round(4).to_string(index=False))

    #Cross effect of H and T on a grid
    grid_prices, grid_errors = grid_sweep(sweep_sim.S)
    print(f"\nUOBC price on the (T, H) grid ({N_PATHS_SWEEP} paths), rows T, columns H/S0")
    print(grid_prices.round(2).to_string())
    print(f"Largest std error on the grid: {grid_errors.values.max():.2f}")

    #does the price go up with H for every T, and for which T is the price highest for each H?
    increases_with_H = (grid_prices.diff(axis=1).iloc[:, 1:] >= 0).all(axis=1)
    print(f"Price increases with H for every T: {increases_with_H.all()}")
    print("Maturity with the highest price, for each H/S0:")
    print(grid_prices.idxmax(axis=0).to_string())

    #The CSV Files are o that we can make the report more easily and ust copy paste the values into the tables
    barrier_table.to_csv(RESULTS_DIR / "uobc_barrier.csv", index=False)
    maturity_table.to_csv(RESULTS_DIR / "uobc_maturity.csv", index=False)
    grid_prices.to_csv(RESULTS_DIR / "uobc_grid_prices.csv")
    grid_errors.to_csv(RESULTS_DIR / "uobc_grid_std_errors.csv")
    plot_barrier_sweep(barrier_table, call_price)
    plot_maturity_sweep(maturity_table)
    plot_grid(grid_prices)

    plt.show()
