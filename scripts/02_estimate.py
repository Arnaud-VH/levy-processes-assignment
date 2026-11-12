"""
Part 2 of project.
Loads the option data from the CSV and appropriate functions from param_estimation.py to estimate the model parameters.
Save those parameters in a file for the later parts. 
"""
import csv
import numpy as np
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import ndtr
from scipy.optimize import minimize_scalar
from scipy.integrate import simpson
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "Call_data.xlsx"


# Market Parameters — 31/03/2026
S0 = 6528.52               
q = 0.012856               
r = 0.0365529             
REFERENCE_DATE = pd.Timestamp("2026-03-31")

# Step (in simple returns r_k) per maturity, for the first four maturities.
# T=17/365 -> 1.0% ; T=45/365 -> 1.75% ; T=79/365 -> 2.5% ; T=108/365 -> 3.25%
STEPS = {
    17:  0.01,
    45:  0.0175,
    79:  0.025,
    108: 0.0325,
}

# Special cases
T_SPECIAL_NONUNIFORM = 143   # 21/08/2026
T_SPECIAL_FOUR = 171   # 18/09/2026

N_STRIKES        = 15 #strikes per maturity
N_STRIKES_LAST   = 4 #strike for the last maturity

def forward_price(T):
    return S0 * math.exp((r - q) * T)

# Data loading
class DataHandler:
    def __init__(self, path=DATA_FILE):
        self.df = self._load_data(path)
        self.df_calls = self._load_calls(path)

    @staticmethod
    def _load_data(path):
        df = pd.read_excel(path, sheet_name="Data")
        return df.set_index("Date")

    @staticmethod
    def _load_calls(path):
        # Read only the only_call_SPXW sheet.
        # Expected columns (by position, to avoid issues with the ArrayFormula in column A): [ticker, PX_Last, Exp_date, Strike, Type]
        raw = pd.read_excel(path, sheet_name="only_call_SPXW")
        raw = raw.iloc[:, [1, 2, 3]]
        raw.columns = ["price", "maturity", "strike"]

        raw["maturity"] = pd.to_datetime(raw["maturity"], dayfirst=True)
        raw = raw[raw["price"].notna() & (raw["price"] > 0)].copy()
        raw["T"] = (raw["maturity"] - REFERENCE_DATE).dt.days / 365.0

        return raw[["maturity", "T", "strike", "price"]].reset_index(drop=True)

# Grid construction
def _greedy_closest(targets, available):
    # This method consist in a local heuristic that for each target k, returns the closest available strike not yet included -> uniqueness/non duplicate points
    available = sorted(available)
    used = set()
    picked = []
    for t in targets:
        best = None
        best_distance = float("inf")
        for k in available:
            if k in used:
                continue
            d = abs(k - t) #definiton of absolute module distance
            if d < best_distance:
                best_distance = d
                best = k
        picked.append(best)
        used.add(best)
    return picked

def build_uniform_grid(available, F_T, step, n=N_STRIKES):
    # Uniform grid in simple returns r_k = (K - S0)/S0, centred on the available strike closest to F(T). Returns n distinct strikes.
    strikes = sorted(available)
    k_center = min(strikes, key=lambda k: abs(k - F_T))
    half = n // 2 #because we impose all the uniform grids to be odd
    targets = [k_center + j * step * S0 for j in range(-half, half + 1)]
    return sorted(_greedy_closest(targets, strikes))

#contruction of the Grid for 21/08/2026
# strategy: 1) the n_middle strikes closest to F(T) form the dense ATM block; 2) In each tail, take every other strike (skip one, keep one), closest first. 3) Trim or fill to exactly the dersirable n strikes.
def build_tail_skip_grid(available, F_T, n=N_STRIKES, n_middle=7):
    by_dist = sorted(available, key=lambda k: abs(k - F_T)) #sorted in terms of the distance to the center (Forward)
    middle = sorted(by_dist[:n_middle]) #slices the n_middle points close to the center -> it is our ATM dense Block
    itm = sorted([k for k in available if k < middle[0]],
                 key=lambda k: abs(k - F_T))
    otm = sorted([k for k in available if k > middle[-1]],
                 key=lambda k: abs(k - F_T))
    #jumps: "um sim, um nao"
    itm_sel = itm[::2] 
    otm_sel = otm[::2]
    chosen = middle + itm_sel + otm_sel
    if len(chosen) > n: #force us to end up with exactly n strikes, as intended
        keep_otm = n - len(middle) - len(itm_sel)
        chosen = middle + itm_sel + otm_sel[:keep_otm]
    elif len(chosen) < n:
        fill = [k for k in otm + itm if k not in chosen]
        for k in fill:
            if len(chosen) >= n:
                break
            chosen.append(k)
    return sorted(chosen) #note that the concatenation above may not preserve the order

#contruction of the Grid for 18/09/2026
def build_four_strikes(available):
    strikes = sorted(available)
    return [strikes[2], strikes[4], strikes[6], strikes[8]]

#CALIBRATION OF THE GRID
def build_calibration_grid(handler):
    # Returns a DataFrame with columns:
    # maturity, T, strike, price, r_k
    calls = handler.df_calls
    rows = []

    # First four maturities: uniform grid
    for days, step in STEPS.items():
        T = days / 365.0
        sub = calls[(calls["T"] - T).abs() < 1e-6]
        if sub.empty:
            continue
        sub = sub.drop_duplicates(subset="strike", keep="first")
        F_T = forward_price(T)
        chosen = build_uniform_grid(sub["strike"].tolist(), F_T, step)

        for K in chosen:
            row = sub[sub["strike"] == K].iloc[0]
            rows.append(dict(maturity=row["maturity"], T=row["T"], strike=float(K), price=float(row["price"])))

    # Grid for 21/08/2026
    T = T_SPECIAL_NONUNIFORM / 365.0
    sub = calls[(calls["T"] - T).abs() < 1e-6]
    if not sub.empty:
        sub = sub.drop_duplicates(subset="strike", keep="first")
        F_T = forward_price(T)
        chosen = build_tail_skip_grid(sub["strike"].tolist(), F_T)
        for K in chosen:
            row = sub[sub["strike"] == K].iloc[0]
            rows.append(dict(maturity=row["maturity"], T=row["T"], strike=float(K), price=float(row["price"])))

    #Grid for 18/09/2026
    T = T_SPECIAL_FOUR / 365.0
    sub = calls[(calls["T"] - T).abs() < 1e-6]
    if not sub.empty:
        sub = sub.drop_duplicates(subset="strike", keep="first")
        chosen = build_four_strikes(sub["strike"].tolist())
        for K in chosen:
            row = sub[sub["strike"] == K].iloc[0]
            rows.append(dict(maturity=row["maturity"], T=row["T"], strike=float(K), price=float(row["price"])))

    grid = pd.DataFrame(rows) #grid data
    grid["r_k"] = (grid["strike"] - S0) / S0 #grid data in returns
    return grid

#function that stores the grid data into a dictionary
def grid_to_dict(grid):
    #dictionary {(T, K): price}; T rounded to 8 decimal places.
    result = {}
    for _, row in grid.iterrows():
        key = (round(float(row["T"]), 8), float(row["strike"]))
        value = float(row["price"])
        result[key] = value
    return result

#function that constructs a table that compacts all the information regarding the chosen dataset <-> similar to the Appendix C of Schoutens
def print_apx_c_matrix(grid):
    #rows = strikes (with r_k % in parentheses), columns = maturity dates
    pivot = grid.pivot_table(index="strike", columns="maturity", values="price")
    pivot.columns = [c.strftime("%d/%m/%Y") for c in pivot.columns] #converts the data to be presented in the standard format day/month/year
    #build strike labels with the simple-return percentage
    r_k = (pivot.index - S0)/S0
    strike_labels = [f"{int(k)} ({v*100:+.2f}%)" for k,v in zip(pivot.index, r_k)]
    pivot.index = strike_labels
    #round prices to 2 decimals and leave missing cells blank
    pivot = pivot.round(2).fillna("")
    print(pivot.to_string())

#using matplotlib.pyplot library to produce the same table but as a PNG
def save_apx_c_matrix_as_png(grid, filename="sp500_table.png"):
    pivot = grid.pivot_table(index="strike", columns="maturity", values="price")
    pivot.columns = [c.strftime("%d/%m/%Y") for c in pivot.columns]
    r_k = (pivot.index - S0) / S0
    pivot.index = [f"{int(k)} ({v*100:+.2f}%)" for k, v in zip(pivot.index, r_k)]
    pivot = pivot.round(2).fillna("")

    fig, ax = plt.subplots(figsize=(9, 14))
    ax.axis("off")

    # Título ajustado (x=0.42)
    ax.set_title("S&P 500 Call Option Prices",
                 fontsize=14, fontweight="bold", pad=4, x=0.42)

    # Tabela colada ao topo (98% da altura)
    table = ax.table(
        cellText=pivot.values,
        rowLabels=pivot.index,
        colLabels=pivot.columns,
        cellLoc="right",
        rowLoc="right",
        bbox=[0.0, 0.0, 1.0, 0.98],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    for (row, col), cell in table.get_celld().items():
        # Remove todas as grelhas
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        cell.set_edgecolor("white")
        
        # Cabeçalho e coluna dos strikes em bold
        if row == 0 or col == -1:
            cell.set_text_props(weight="bold")

    plt.savefig(filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {filename}")



handler = DataHandler()
grid = build_calibration_grid(handler)


K = grid["strike"].to_numpy(dtype=float)
T = grid["T"].to_numpy(dtype=float)
r = np.full(len(K), 0.0365529)
market = grid["price"].to_numpy(dtype=float)


u = np.linspace(1e-6, 500,2001)

def eta_VG(u,mu,sigma,a):
    return -a * np.log(1 - 1j*mu*u/a + sigma**2*u**2/(2*a))

def phi_CIR(t, z, kappa, lam, y0):
    gamma = np.sqrt(kappa**2 + 2*lam**2*z + 0j)
    d = (gamma+kappa)*(1- np.exp(-gamma*t)) + 2*gamma* np.exp(-gamma*t)
    return np.exp((2*kappa/lam**2)*np.log(2*gamma*np.exp((kappa-gamma)*t/2)/d) - 2*z*(1- np.exp(-gamma*t))/d*y0)

def phi_VGCIR(u, t, mu, sigma, a, kappa, lam, y0):
    return phi_CIR(t, -eta_VG(u, mu, sigma, a), kappa, lam, y0)


def phi_logS(u, t, r, mu, sigma, a, kappa, lam, y0):
    z = 1j*u*eta_VG(-1j, mu, sigma, a) - eta_VG(u, mu, sigma, a)
    return np.exp(1j*u*(np.log(S0)+(r-q)*t)) * phi_CIR(t, z, kappa, lam, y0)


def call_price(K, t, r, mu, sigma, a, kappa, lam, y0):
    params = (mu, sigma, a, kappa, lam, y0)
    def integrand1(u):
        return np.real(
            np.exp(-1j*u*np.log(K)) *
            phi_logS(u-1j, t, r, *params) /
            (1j*u*phi_logS(-1j, t, r, *params))
        )
    def integrand2(u):
        return np.real(
            np.exp(-1j*u*np.log(K)) *
            phi_logS(u, t, r, *params) /
            (1j*u)
        )
    P1 = 0.5 + simpson(integrand1(u), x=u)/np.pi
    P2 = 0.5 + simpson(integrand2(u), x=u)/np.pi

    return S0*np.exp(-q*t)*P1 - K*np.exp(-r*t)*P2

def residuals(params):
    mu, sigma, a, kappa, lam, y0 = params

    if 1 - mu/a - sigma**2/(2*a) <= 0:
        return np.full(len(K), 1e4)

    model = np.array([
        call_price(K[i], T[i], r[i], *params)
        for i in range(len(K))
    ])

    if not np.all(np.isfinite(model)):
        return np.full(len(K), 1e4)

    return (model - market)/market


params = (0.1, 0.2, 1, 2, 0.5, 1)

errors = residuals(params)

print(errors[:5])
print("Sum of squared relative errors:", np.sum(errors**2))


# VG-CIR calibration

lower = [-5, 1e-4, 0.01, 0.01, 0.01, 0.001]
upper = [5, 5, 200, 200, 50, 50]

initial_guesses = [
    [0, 0.1, 50, 10, 4, 5],
]

best = None

for initial in initial_guesses:
    result = least_squares(residuals,initial,bounds=(lower, upper),x_scale="jac",max_nfev=500)
    if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
        best = result

result = best

print("Estimated VG-CIR parameters:", result.x)


# VG-CIR ARPE
vg_prices = np.array([
    call_price(K[i], T[i], r[i], *result.x)
    for i in range(len(K))
])

vg_arpe = 100*np.mean(np.abs((vg_prices-market)/market))



# Variance Gamma benchmark

def phi_logS_VG(u, t, r, mu, sigma, a):
    correction = eta_VG(-1j, mu, sigma, a)

    return np.exp(
        1j*u*(np.log(S0)+(r-q)*t)
        + t*(eta_VG(u, mu, sigma, a)-1j*u*correction)
    )


def call_price_VG(K, t, r, mu, sigma, a):

    params = (mu, sigma, a)

    def integrand1(u):
        return np.real(np.exp(-1j*u*np.log(K)) *phi_logS_VG(u-1j, t, r, *params) /(1j*u*phi_logS_VG(-1j, t, r, *params)))

    def integrand2(u):
        return np.real(np.exp(-1j*u*np.log(K)) *phi_logS_VG(u, t, r, *params) /(1j*u))

    P1 = 0.5 + simpson(integrand1(u), x=u)/np.pi
    P2 = 0.5 + simpson(integrand2(u), x=u)/np.pi

    return S0*np.exp(-q*t)*P1 - K*np.exp(-r*t)*P2


def residuals_VG(params):
    mu, sigma, a = params

    if 1 - mu/a - sigma**2/(2*a) <= 0:
        return np.full(len(K), 1e4)

    model = np.array([
        call_price_VG(K[i], T[i], r[i], *params)
        for i in range(len(K))
    ])

    if not np.all(np.isfinite(model)):
        return np.full(len(K), 1e4)

    return (model-market)/market


lower_VG = [-5, 1e-4, 0.01]
upper_VG = [5, 5, 200]

initial_guesses_VG = [
    [0, 0.15, 5],
]

best_VG = None

for initial in initial_guesses_VG:

    result_VG = least_squares(
        residuals_VG,
        initial,
        bounds=(lower_VG, upper_VG),
        x_scale="jac",
        max_nfev=500
    )

    if best_VG is None or np.sum(result_VG.fun**2) < np.sum(best_VG.fun**2):
        best_VG = result_VG

estimated_VG = best_VG.x

prices_VG = np.array([
    call_price_VG(K[i], T[i], r[i], *estimated_VG)
    for i in range(len(K))
])

arpe_VG = 100*np.mean(
    np.abs((prices_VG-market)/market)
)


# Black-Scholes benchmark



def bs_prices(vol):
    v = vol*np.sqrt(T)

    d1 = (np.log(S0/K) + (r-q+0.5*vol**2)*T)/v
    d2 = d1-v

    return S0*np.exp(-q*T)*ndtr(d1) - K*np.exp(-r*T)*ndtr(d2)


def bs_loss(vol):
    return np.sum(((bs_prices(vol)-market)/market)**2)


bs_fit = minimize_scalar(bs_loss,bounds=(0.001, 5),method="bounded")

bs_prices_fitted = bs_prices(bs_fit.x)

bs_arpe = 100*np.mean(np.abs((bs_prices_fitted-market)/market))


# Comparison

print("\nEstimated VG-CIR parameters:", result.x)
print("Estimated VG parameters:", estimated_VG)

print("\nVG-CIR ARPE:       ", round(vg_arpe, 3), "%")
print("Variance Gamma ARPE:", round(arpe_VG, 3), "%")
print("Black-Scholes ARPE:", round(bs_arpe, 3), "%")


