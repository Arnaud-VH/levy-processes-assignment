"""
Part 4 of the project.
Monte Carlo pricing of the up-and-out barrier call option.

The pricer is model-free: it only needs simulated stock paths S with shape (n_paths, n_steps + 1),
so the same classes work with the VG-CIR paths from simulation.py or any other model.
A new option only needs a payoff(S) method, see LookbackCall at the bottom.
"""

from dataclasses import dataclass

import numpy as np

@dataclass
class MCPrice:
    """
    Output of Option.price. The price and the Monte Carlo standard error,
    so we can report a confidence interval: price +/- 1.96 * std_error.
    """
    price: float
    std_error: float

class Option:
    """
    Base class for an option priced by Monte Carlo.
    Subclasses only have to implement payoff(S), where S is the path array, the rest is shared.
    Only the maturity is stored here, because that is the only thing every option has (a lookback call has no strike).
    """

    def __init__(self, maturity):
        if maturity <= 0:
            raise ValueError(f"maturity must be positive, got {maturity}")
        self.maturity = maturity

    def payoff(self, S):
        """
        Payoff at maturity for every path.
        S has shape (n_paths, n_steps + 1), the result has shape (n_paths,).
        """
        raise NotImplementedError("Subclasses must implement payoff(S).")

    def price(self, S, market):
        """
        Risk-neutral price: discounted sample mean of the payoff.
        The standard error is std / sqrt(n_paths), so it shrinks with more paths.
        """
        discounted = np.exp(-market.r * self.maturity) * self.payoff(S)
        n_paths = len(discounted)
        return MCPrice(price=discounted.mean(), std_error=discounted.std(ddof=1) / np.sqrt(n_paths))

class EuropeanCall(Option):
    """
    Plain vanilla call: max(S_T - K, 0).
    Used to check the Monte Carlo pricer against the analytical pricer (analytical.py).
    """

    def __init__(self, strike, maturity):
        super().__init__(maturity)
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")
        self.strike = strike

    def payoff(self, S):
        S_T = S[:, -1]
        return np.maximum(S_T - self.strike, 0.0)

class UpAndOutCall(EuropeanCall):
    """
    Up-and-out barrier call (UOBC): pays like a plain call unless the stock
    touches or goes above the barrier H at some point before maturity, in which case it pays 0.
    Schoutens (2003) Section 9.3, exotic options.

    The barrier is checked on the simulation grid only (discrete monitoring),
    so with few time steps we slightly underestimate the chance of knocking out
    and overestimate the price.
    """

    def __init__(self, strike, maturity, barrier):
        super().__init__(strike, maturity)
        if barrier <= strike:
            raise ValueError(f"barrier must be above the strike, got H={barrier} and K={strike}")
        self.barrier = barrier

    def knocked_out(self, S):
        """True for the paths whose running maximum reaches the barrier."""
        return S.max(axis=1) >= self.barrier

    def payoff(self, S):
        plain_call = super().payoff(S)
        return np.where(self.knocked_out(S), 0.0, plain_call)

class LookbackCall(Option):
    """
    Floating strike lookback call: S_T - min of S over [0, T], so the holder buys at the lowest price of the path.
    Schoutens (2003) Section 9.1, priced for VG-CIR in Table 9.1 (LC).
    Only a payoff is needed, pricing and the standard error come from Option.

    Like the barrier, the minimum is only checked on the simulation grid, so the simulated minimum
    is a bit too high and the price a bit too low compared to watching the path continuously.
    """

    def payoff(self, S):
        return S[:, -1] - S.min(axis=1)
