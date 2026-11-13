"""
Part 4 extension.
Compares the up-and-out barrier call under VG-CIR (Monte Carlo) with Black-Scholes (closed form, Schoutens p. 121-122).
1. Checks our Monte Carlo barrier code against the exact Black-Scholes price.
2. Computes the same (T, H) grid as script 04 with the Black-Scholes formula, to check the shape of the VG-CIR grid.
3. Compares the barrier prices of the two models (model risk). Black-Scholes is used with two volatilities:
   the one fitted to all our call prices in Part 2, and the one that gives the same 9 month call price as VG-CIR (implied volatility).
   With the second one both models agree on the plain call, so any difference in the UOBC comes from the barrier.
Run from the repo root: python scripts/05_black_scholes_barrier.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vg_cir.params import BS_SIGMA, ESTIMATED_VGCIR, DATASET_MARKET
from vg_cir.pricing.analytical import call_price as vgcir_call
from vg_cir.pricing.black_scholes import bs_call, bs_up_and_out_call, discrete_barrier, implied_vol, simulate_bs
from vg_cir.pricing.montecarlo import UpAndOutCall
from vg_cir.simulation import simulate_vgcir

#Folder of the repo (one above scripts/), so the outputs go to the same results/ wherever the script is run from
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"

PARAMS = ESTIMATED_VGCIR
MARKET = DATASET_MARKET
SIGMA = BS_SIGMA

#Same setup as script 04
T = 0.75
STRIKE = MARKET.S0
BARRIER = 1.25 * MARKET.S0
N_STEPS = 270
DT = T / N_STEPS
SEED = 42
N_PATHS_SWEEP = 20_000
H_RATIOS = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.60, 1.80]
MATURITIES = [0.10, 0.25, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]

#More paths for the check against the formula, Black-Scholes paths are cheap to simulate
N_PATHS_CHECK = 100_000


def bs_uobc(maturity, H, sigma=SIGMA):
    """Black-Scholes UOBC for a barrier checked every DT, like in our simulation (shifted barrier)."""
    return bs_up_and_out_call(STRIKE, maturity, discrete_barrier(H, sigma, DT), sigma, MARKET)


def check_mc_against_formula():
    """
    Monte Carlo on Black-Scholes paths with our UpAndOutCall class, against the exact formula.
    The formula is for a barrier checked continuously, so we compare with the shifted barrier,
    and also show the unshifted price to see how much the discrete checks matter.
    """
    S = simulate_bs(SIGMA, MARKET, T, N_STEPS, N_PATHS_CHECK, seed=SEED)
    rows = []
    for h in [1.10, 1.25, 1.40]:
        H = h * MARKET.S0
        mc = UpAndOutCall(STRIKE, T, H).price(S, MARKET)
        exact = bs_uobc(T, H)
        rows.append({"H/S0": h, "MC": mc.price, "std_error": mc.std_error,
                     "formula (checked every dt)": exact,
                     "difference in std errors": (mc.price - exact) / mc.std_error,
                     "formula (checked continuously)": bs_up_and_out_call(STRIKE, T, H, SIGMA, MARKET)})
    return pd.DataFrame(rows)


def bs_grid():
    """Black-Scholes UOBC for every pair (T, H) in MATURITIES x H_RATIOS. Rows T, columns H/S0."""
    prices = pd.DataFrame(index=MATURITIES, columns=H_RATIOS, dtype=float)
    for maturity in MATURITIES:
        for h in H_RATIOS:
            prices.loc[maturity, h] = bs_uobc(maturity, h * MARKET.S0)
    prices.index.name = "T"
    prices.columns.name = "H/S0"
    return prices


def compare_models(sigma_matched):
    """
    UOBC under VG-CIR (Monte Carlo, same paths as script 04) and Black-Scholes (formula),
    along H with T = 9 months and along T with H = 1.25 S0.
    Along H we also use Black-Scholes with sigma_matched, which gives the same 9 month call as VG-CIR.
    """
    t_max = max(MATURITIES)
    S = simulate_vgcir(PARAMS, MARKET, t_max, round(t_max / DT), N_PATHS_SWEEP, seed=SEED).S

    along_H = []
    S_T = S[:, :round(T / DT) + 1]
    for h in H_RATIOS:
        H = h * MARKET.S0
        along_H.append({"H/S0": h, "VG-CIR": UpAndOutCall(STRIKE, T, H).price(S_T, MARKET).price,
                        "Black-Scholes": bs_uobc(T, H),
                        "Black-Scholes (matched)": bs_uobc(T, H, sigma_matched)})

    along_T = []
    for maturity in MATURITIES:
        S_cut = S[:, :round(maturity / DT) + 1]
        along_T.append({"T": maturity, "VG-CIR": UpAndOutCall(STRIKE, maturity, BARRIER).price(S_cut, MARKET).price,
                        "Black-Scholes": bs_uobc(maturity, BARRIER)})
    return pd.DataFrame(along_H), pd.DataFrame(along_T)


def plot_models(along_H, along_T, sigma_matched):
    """UOBC under both models, against H (left) and against T (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model, style, label in [("VG-CIR", "o-", "VG-CIR"), 
                                ("Black-Scholes", "s--", f"Black-Scholes, fitted $\\sigma$ = {SIGMA:.3f}")]:
        axes[0].plot(along_H["H/S0"], along_H[model], style, label=label)
        axes[1].plot(along_T["T"], along_T[model], style, label=label)
    axes[0].plot(along_H["H/S0"], along_H["Black-Scholes (matched)"], "^:",
                 label=f"Black-Scholes, same 9m call ($\\sigma$ = {sigma_matched:.3f})")
    axes[0].set_title(f"UOBC against the barrier (T = {T})")
    axes[0].set_xlabel("$H / S_0$")
    axes[1].set_title("UOBC against the maturity (H = 1.25 $S_0$)")
    axes[1].set_xlabel("T (years)")
    for ax in axes:
        ax.set_ylabel("price")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "uobc_bs_vs_vgcir.png", dpi=150)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    #Our Monte Carlo barrier code against the exact Black-Scholes price
    check = check_mc_against_formula()
    print(f"1. Monte Carlo vs formula on Black-Scholes paths (T = {T}, {N_STEPS} steps, {N_PATHS_CHECK} paths, sigma = {SIGMA:.4f})")
    print(check.round(4).to_string(index=False))

    #Shape of the grid under Black-Scholes
    grid = bs_grid()
    print("\n2. Black-Scholes UOBC on the (T, H) grid, barrier checked every dt, rows T, columns H/S0")
    print(grid.round(2).to_string())
    increases_with_H = (grid.diff(axis=1).iloc[:, 1:] >= 0).all(axis=1)
    print(f"Price increases with H for every T: {increases_with_H.all()}")
    print("Maturity with the highest price, for each H/S0:")
    print(grid.idxmax(axis=0).to_string())

    #The two models side by side
    vg_call = vgcir_call(STRIKE, T, PARAMS, MARKET)
    sigma_matched = implied_vol(vg_call, STRIKE, T, MARKET)
    along_H, along_T = compare_models(sigma_matched)
    vg_uobc = along_H.loc[along_H["H/S0"] == 1.25, "VG-CIR"].item()
    print(f"\n3. Assignment setup (T = {T}, K = S0, H = 1.25 S0)")
    print(f"{'':<14} {'VG-CIR':>10} {'BS fitted':>12} {'BS matched':>12}")
    print(f"{'sigma':<14} {'':>10} {SIGMA:12.4f} {sigma_matched:12.4f}")
    print(f"{'Plain call':<14} {vg_call:10.2f} {bs_call(STRIKE, T, SIGMA, MARKET):12.2f} {bs_call(STRIKE, T, sigma_matched, MARKET):12.2f}")
    print(f"{'UOBC':<14} {vg_uobc:10.2f} {bs_uobc(T, BARRIER):12.2f} {bs_uobc(T, BARRIER, sigma_matched):12.2f}")
    print("\nAlong H:")
    print(along_H.round(2).to_string(index=False))
    print("\nAlong T:")
    print(along_T.round(2).to_string(index=False))

    #again using the CSVs to store the data so we can put it into the report more easily
    check.to_csv(RESULTS_DIR / "bs_mc_check.csv", index=False)
    grid.to_csv(RESULTS_DIR / "bs_grid_prices.csv")
    along_H.to_csv(RESULTS_DIR / "bs_vs_vgcir_barrier.csv", index=False)
    along_T.to_csv(RESULTS_DIR / "bs_vs_vgcir_maturity.csv", index=False)

    plot_models(along_H, along_T, sigma_matched)

    plt.show()
