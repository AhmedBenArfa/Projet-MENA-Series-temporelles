"""Classical forecasters: ARIMA & SARIMA walk-forward (train once, no refit).

Both functions fit a statsmodels SARIMAX model once on the training set,
then roll forward one step at a time across the test set using
`.append(new_obs, refit=False)` so the model parameters are never
re-estimated. Every step's one-step-ahead mean forecast is captured before
the new observation is appended.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tsvar.var import ForecastResult


def _fit(train, order, seasonal_order):
    m = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order or (0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return m.fit(disp=False, maxiter=200, method="lbfgs")


def walk_forward_arima(train, test, order=None, seasonal_order=None) -> ForecastResult:
    if order is None:
        import pmdarima as pm

        order = pm.auto_arima(
            train.values, seasonal=False, suppress_warnings=True, error_action="ignore"
        ).order

    # Use a plain integer index internally so statsmodels never infers a date
    # frequency (train and test come from two separate CSVs and may not abut).
    n = len(train)
    train_ri = pd.Series(np.asarray(train.values), index=pd.RangeIndex(n))
    res = _fit(train_ri, order, seasonal_order)

    # res.resid includes burn-in artifacts at index 0; drop it, then drop any
    # remaining non-finite values so the residual pool is always usable.
    raw_resid = np.asarray(res.resid)[1:]
    raw_resid = raw_resid[np.isfinite(raw_resid)]
    sigma = float(np.std(raw_resid))
    std_resid = (raw_resid - np.mean(raw_resid)) / sigma

    mu = np.empty(len(test))
    cur = res
    for t in range(len(test)):
        mu[t] = float(cur.forecast(1).iloc[0])
        nxt = pd.Series([test.values[t]], index=pd.RangeIndex(n + t, n + t + 1))
        cur = cur.append(nxt, refit=False)  # walk forward, no re-estimation

    return ForecastResult(
        mu=mu,
        sigma=np.full(len(test), sigma),
        std_resid=np.asarray(std_resid),
        y_true=test.values,
        dates=test.index,
        name=getattr(test, "name", None) or "series",
    )


def walk_forward_sarima(train, test, m=5) -> ForecastResult:
    return walk_forward_arima(train, test, order=(1, 0, 1), seasonal_order=(1, 0, 1, m))
