# levy-processes-assignment
Group assignment for Levy Processes course in the MSc in Mathematical Finance at ISEG Lisbon

## Structure
Files in the `scripts` directory call the functions from our "library" `src/vg_cir` and use the appropriate parameters and data.

The files in `src/vg_cir` are written as re-usable, modular functions (parameters, characteristic functions, simulation, pricers), used across the different parts of the assignment and by the tests.

- `data/Call_data.xlsx`: S&P 500 data and SPXW call prices on 31/03/2026.
- `results/`: CSV tables and figures written by the scripts.
- `tests/`: automatic checks of the simulation and the pricers.

## Note
The estimated parameters are stored by hand in `src/vg_cir/params.py`. Scripts 03-07 read them from there, so they don't need 01 or 02 to run first. If you re-run 02 and want to use new estimates, copy them into `params.py`.

## Set-up
From the repo root `levy-processes-assignment` type:
- `python -m pip install -e .`
- `python -m pip install pytest`

## How to run
All commands below are run from the project root `levy-processes-assignment`.

### Part 1: dataset
Loads the Excel data and builds the calibration grid (79 calls), then prints it.
```
python scripts/01_dataset.py
```

### Part 2: parameter estimation
Calibrates VGSA, VGSAM, Variance Gamma and Black-Scholes, and prints the parameters and the ARPE. Takes about 4 minutes. Saves `pricing_errors_by_maturity.png` and `.pdf` in the folder you run it from.
```
python scripts/02_estimate.py
```

### Part 3: simulation of the paths
VG-CIR paths, distribution of the returns and moments. Figures go to `results/figures/`. For the parameter sensitivity analysis set `RUN_SENSITIVITY = True` in the file (slow, figures go to `results/figures/sensitivity/`).
```
python scripts/03_simulate_paths.py
```
Benchmarks for the comparison, Black-Scholes and Variance Gamma paths and return distribution (`results/figures/bs_*.png`, `results/figures/vg_*.png`):
```
python scripts/06_black_scholes_paths.py
python scripts/07_variance_gamma_paths.py
```

### Part 4: pricing of the up-and-out barrier call
Monte Carlo price of the UOBC (T = 0.75, K = S0, H = 1.25 S0) and the effect of H and T. Tables go to `results/uobc_*.csv`, figures to `results/figures/`.
```
python scripts/04_price_barrier.py
```
Check of the barrier code against the Black-Scholes formula and comparison of the two models. Tables go to `results/bs_*.csv`, figures to `results/figures/`.
```
python scripts/05_black_scholes_barrier.py
```

### Tests
```
python -m pytest
```
