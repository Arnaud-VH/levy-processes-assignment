"""
Data class for the model and market parameters of the VG-CIR model. 
Default model parameter values are taken from the Schoutens book Table 7.3 on page 96. 
Market model parameters are also taken from the book. 
"""

from dataclasses import dataclass

@dataclass
class VGCIRParams:
    """
    Class for the risk-neutral parameters of the VG-CIR model.  
    """
    C: float
    G: float
    M: float
    kappa: float
    eta: float
    lam: float
    y0: float = 1.0

    def __post_init__(self):
        for param in ("C","G", "M", "kappa", "eta", "lam", "y0"):
            if getattr(self, param) < 0:
                raise ValueError(f"{param} must be non-negative, instead function got {getattr(self, param)}.")
        if self.M <= 1:
            #Check from Schoutens book to ensure forward stock price is not infinite. 
            raise ValueError(f"M must be > 1 for the martingale correction to work. Got {self.M} instead.")

@dataclass 
class MarketParams:
    """
    Class for the market inputs. 
    Spot, Risk-free rate and dividend yield. 
    """
    S0: float
    r: float
    q: float

    def __post_init__(self):
        if self.S0 <= 0:
            raise ValueError(f"S0 must be positive, got {self.S0}")

BOOK_VGCIR = VGCIRParams(C=11.9896, G=25.8523, M=35.5344, kappa=0.6020, eta=1.5560, lam=1.9992, y0=1.0)

BOOK_MARKET = MarketParams(S0=1124.47, r=0.019, q=0.012)

#Estimated parameters of the VGSAM model (martingale correction along the clock), Table 2 of the report (Part 2).
ESTIMATED_VGCIR = VGCIRParams(C=24.007776, G=15.613156, M=124.027527, kappa=31.120776, eta=0.290767, lam=3.921757, y0=1.0)

#Black-Scholes volatility fitted to the same call prices in 02_estimate.py. Used to compare in part 4

BS_SIGMA = 0.14388654

#Our dataset (Part 1): S&P 500 on 31/03/2026, same r and q as used in the calibration (02_estimate.py)
DATASET_MARKET = MarketParams(S0=6528.52, r=0.0365529, q=0.012856)
