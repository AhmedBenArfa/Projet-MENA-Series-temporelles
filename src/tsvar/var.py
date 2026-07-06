"""Bootstrap Historical Simulation (BHS) VaR engine.

Defines the shared ForecastResult interface produced by every forecasting
model in this project, and the BHS quantile function that turns a
location-scale forecast (mu, sigma) plus a pool of standardized residuals
into a Value-at-Risk return level.

Sign convention: all functions here return the alpha-quantile of the
predictive RETURN distribution (left tail, typically negative). A VaR
violation is defined elsewhere as `realized_return < var_return`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tsvar import SEED


@dataclass
class ForecastResult:
    mu: np.ndarray
    sigma: np.ndarray
    std_resid: np.ndarray
    y_true: np.ndarray
    dates: pd.DatetimeIndex
    name: str


def bhs_quantile(mu, sigma, std_resid, alpha, n_boot=10000, seed=SEED) -> float:
    """Bootstrap Historical Simulation quantile (a return level, left tail).

    mu + sigma * empirical_alpha_quantile(bootstrap_resample(std_resid))
    """
    rng = np.random.default_rng(seed)
    boot = rng.choice(std_resid, size=n_boot, replace=True)
    q = np.quantile(boot, alpha)
    return float(mu + sigma * q)


def var_series(fc: ForecastResult, alpha, n_boot=10000, seed=SEED) -> np.ndarray:
    """Per-day VaR return levels for the test set described by `fc`."""
    return np.array([
        bhs_quantile(fc.mu[t], fc.sigma[t], fc.std_resid, alpha, n_boot, seed + t)
        for t in range(len(fc.mu))
    ])
