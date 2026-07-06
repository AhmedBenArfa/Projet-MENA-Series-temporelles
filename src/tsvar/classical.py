"""Classical forecasters: ARIMA & SARIMA walk-forward (train once, no refit).

Both functions fit a statsmodels SARIMAX model once on the training set,
then roll forward one step at a time across the test set using
`.append(new_obs, refit=False)` so the model parameters are never
re-estimated. Every step's one-step-ahead mean forecast is captured before
the new observation is appended.
"""

import numpy as np
import pandas as pd
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
    return m.fit(disp=False)


def walk_forward_arima(train, test, order=None, seasonal_order=None) -> ForecastResult:
    if order is None:
        import pmdarima as pm

        order = pm.auto_arima(
            train, seasonal=False, suppress_warnings=True, error_action="ignore"
        ).order

    res = _fit(train, order, seasonal_order)

    # res.resid includes burn-in artifacts at index 0; drop it, then drop any
    # remaining non-finite values so the residual pool is always usable.
    raw_resid = np.asarray(res.resid)[1:]
    raw_resid = raw_resid[np.isfinite(raw_resid)]
    sigma = float(np.std(raw_resid))
    std_resid = (raw_resid - np.mean(raw_resid)) / sigma

    # statsmodels' `.append(refit=False)` requires the appended observation's
    # index to *extend* the fitted model's index (same freq, next timestamp).
    # `test` is an independent series with its own (possibly non-contiguous
    # or overlapping) index, so we build a shadow index that continues
    # directly from `train`'s index at `train`'s frequency purely for the
    # walk-forward bookkeeping. The original `test.index` is still what gets
    # returned in `dates` below.
    train_index = train.index
    if isinstance(train_index, pd.DatetimeIndex) and train_index.freq is not None:
        continuation_index = pd.date_range(
            start=train_index[-1], periods=len(test) + 1, freq=train_index.freq
        )[1:]
    else:
        continuation_index = pd.RangeIndex(len(train_index), len(train_index) + len(test))
    walk_test = pd.Series(test.values, index=continuation_index, name=test.name)

    mu = np.empty(len(test))
    cur = res
    for t in range(len(test)):
        mu[t] = float(cur.forecast(1).iloc[0])
        cur = cur.append(walk_test.iloc[t : t + 1], refit=False)

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
