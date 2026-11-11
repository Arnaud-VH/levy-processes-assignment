from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "Call_data.xlsx"

REFERENCE_DATE = pd.Timestamp("2026-03-31")

# Step (in simple returns r_k) per maturity, for the first four maturities.
# T=17/365 -> 1.0% ; T=45/365 -> 1.75% ; T=79/365 -> 2.5% ; T=108/365 -> 3.25%
STEPS = {
    17:  0.01,
    45:  0.0175,
    79:  0.025,
    108: 0.0325,
}

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

data = DataHandler()                    

# Special cases
T_SPECIAL_NONUNIFORM = 143   # 21/08/2026
T_SPECIAL_FOUR = 171   # 18/09/2026

N_STRIKES        = 15 #strikes per maturity
N_STRIKES_LAST   = 4 #strike for the last maturity

def forward_price(T):
    return data.df['SP500'].item() * math.exp((data.df['US_12M_rate'].item() - data.df['SP500_Dividend_yld'].item()) * T)

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
    targets = [k_center + j * step * data.df['SP500'].item() for j in range(-half, half + 1)]
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
    grid["r_k"] = (grid["strike"] - data.df['SP500'].item()) / data.df['SP500'].item() #grid data in returns
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
    r_k = (pivot.index - data.df['SP500'].item())/data.df['SP500'].item()
    strike_labels = [f"{int(k)} ({v*100:+.2f}%)" for k,v in zip(pivot.index, r_k)]
    pivot.index = strike_labels
    #round prices to 2 decimals and leave missing cells blank
    pivot = pivot.round(2).fillna("")
    print(pivot.to_string())
 
    
# main
grid = build_calibration_grid(data)

print("=" * 72)
print("Market Data:")
print(data.df)
print("=" * 72)
print(f"Valid SPXW calls: {len(data.df_calls)}")
print("=" * 72)
print("Selected grid:")
print(grid.to_string(index=False))
print("=" * 72)
print("Count per maturity:")
print(grid.groupby("maturity").size().to_string())
print(f"Total: {len(grid)} call options")
print("=" * 72)

selected = grid_to_dict(grid)
print("Dictionary {(T, K): price}:")
for key, price in selected.items():
    print(f"  {key}: {price}")
print("=" * 72)
print(f"Entries in the dictionary: {len(selected)}")


print("=" * 72)
print("S&P500 Call Option Prices:")
print_apx_c_matrix(grid)