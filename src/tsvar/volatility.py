"""GARCH volatility forecaster — Filtered Historical Simulation.

Fits a constant-mean GARCH(p, q) model ONCE on the training returns, then
propagates the one-step-ahead conditional variance forward over the test
set using the ESTIMATED PARAMETERS held fixed (the GARCH recursion
``var_t = omega + alpha*last_resid**2 + beta*last_var``), updating only the
variance/residual state with each realized test-set return. No refitting
occurs on the test set (train-once constraint).

`sigma` is therefore dynamic (varies day to day); `mu` is the constant
GARCH mean; `std_resid` is the pool of training standardized residuals
used for Filtered Historical Simulation (bootstrapped in `tsvar.var`).
"""

import numpy as np
import pandas as pd
from arch import arch_model

from tsvar.var import ForecastResult


def walk_forward_garch(train, test, p=1, q=1) -> ForecastResult:
    # Fit ONCE on training data (constant-mean GARCH(p,q)). Reset the index
    # to a plain RangeIndex before fitting: `train`/`test` come from
    # `train_test_returns` with a DatetimeIndex that carries no freq, which
    # makes `arch` emit spurious frequency warnings. The estimated
    # parameters and recursion are unaffected by the index used at fit time.
    train_vals = pd.Series(np.asarray(train.values, dtype=float))
    am = arch_model(train_vals, mean="Constant", vol="GARCH", p=p, q=q, dist="normal")
    res = am.fit(disp="off")
    mu_c = float(res.params["mu"])
    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])
    std_resid = np.asarray(res.std_resid.dropna())

    # Initialise recursion state from the end of the training sample
    last_var = float(res.conditional_volatility.iloc[-1]) ** 2
    last_resid = float(train.values[-1] - mu_c)

    mu = np.full(len(test), mu_c)
    sigma = np.empty(len(test))
    for t in range(len(test)):
        var_t = omega + alpha * last_resid ** 2 + beta * last_var  # 1-step cond. variance, fixed params
        sigma[t] = np.sqrt(var_t)
        last_resid = float(test.values[t] - mu_c)  # walk forward on realized return
        last_var = var_t

    return ForecastResult(
        mu=mu,
        sigma=sigma,
        std_resid=std_resid,
        y_true=test.values,
        dates=test.index,
        name=getattr(test, "name", "series"),
    )
