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
            raise ValueError(f"M must be > 1 for the martingle correction to work. Got {self.M} instead.")

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
            raise ValueError(f"S0 cannot be negative, got {self.S0}")

BOOK_VGCIR = VGCIRParams(C=11.9896, G=25.8523, M=35.5344, kappa=0.6020, eta=1.5560, lam=1.9992, y0=1.0)

BOOK_MARKET = MarketParams(S0=1124.47, r=0.019, q=0.012)
