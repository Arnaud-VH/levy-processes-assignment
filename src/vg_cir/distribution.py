"""
Part 3 of the project, sensitivity analysis (Samuel).
Distribution of the simulated terminal log-returns log(S_T / S0): histogram and sample moments.
Used by 03_simulate_paths.py (VG-CIR), 06_black_scholes_paths.py and 07_variance_gamma_paths.py,
so the three models are compared in the same way.
"""

import matplotlib.pyplot as plt
import numpy as np


def log_returns(S, S0):
    """Terminal log-returns log(S_T / S0) of every path."""
    return np.log(S[:, -1] / S0)


def calculate_moments(S, S0):
    """
    Bias corrected sample mean, variance, skewness and kurtosis of the terminal log-returns.
    The kurtosis is not the excess kurtosis, a normal distribution gives 3.
    """
    log_r = log_returns(S, S0)
    n = len(log_r)
    mean_val = np.mean(log_r)

    #Corrected sample variance
    var_val = np.sum((log_r - mean_val)**2) / (n - 1)

    #Corrected sample skewness
    skew_val = (n / ((n - 1) * (n - 2))) * np.sum(((log_r - mean_val) / np.sqrt(var_val))**3)

    #Corrected sample kurtosis
    kurt_val = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * (np.sum((log_r - mean_val)**4) / var_val**2) \
        - (3 * (n - 1)**2) / ((n - 2) * (n - 3)) + 3

    return {"Mean": mean_val, "Variance": var_val, "Skewness": skew_val, "Kurtosis": kurt_val}


def plot_empirical_pdf(S, S0, n_bins=50, title="Empirical Probability Density Function of Returns",
                       fixed_mu=None, fixed_sigma=None):
    """
    Histogram (relative frequencies) of the terminal log-returns with a normal density on top.
    By default the normal has the sample mean and std of these returns. With fixed_mu and fixed_sigma
    the same normal can be drawn on every plot of the sensitivity analysis, so the plots can be compared.
    """
    log_r = log_returns(S, S0)

    #Relative frequencies
    counts, bin_edges = np.histogram(log_r, bins=n_bins)
    relative_frequencies = counts / len(log_r)

    #Bin centers and width for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(bin_centers, relative_frequencies, width=bin_width,
           color='royalblue', alpha=0.75, edgecolor='black', label='Empirical Returns')

    #Normal density for comparison, times the bin width so it is on the same scale as the relative frequencies
    mu = fixed_mu if fixed_mu is not None else np.mean(log_r)
    sigma = fixed_sigma if fixed_sigma is not None else np.std(log_r)
    x = np.linspace(log_r.min(), log_r.max(), 1000)
    normal_pdf = np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(x, normal_pdf * bin_width, 'r--', lw=2, label='Normal PDF Fit')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r"Log-Returns ($R_T = \ln(S_T/S_0)$)", fontsize=12)
    ax.set_ylabel("Relative Frequency", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_moments_table(results, path):
    """
    Saves a table with the moments of every scenario as a picture.
    results is a dict {scenario name: output of calculate_moments}, path is where the png is saved.
    """
    cell_text = []
    for name, m in results.items():
        cell_text.append([name, f"{m['Mean']:.4f}", f"{m['Variance']:.4f}", f"{m['Skewness']:.4f}", f"{m['Kurtosis']:.4f}"])
    columns = ["Scenario", "Mean", "Variance", "Skewness", "Kurtosis"]

    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.4 * len(results)))
    ax.axis('off')
    table = ax.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#CCCCCC')
        if row == 0:
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#EAEAF2')
        else:
            cell.set_facecolor('#F9F9F9' if row % 2 == 0 else 'white')

    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    return fig
