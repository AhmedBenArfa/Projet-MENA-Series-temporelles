"""Machine-learning one-step-ahead forecasters: Random Forest & XGBoost.

Both models are trained ONCE on the training supervised set (lagged
returns -> next return) and then walked forward across the test set,
feeding REALIZED test returns into the lag window at each step (no
retraining). sigma is the constant std of the training residuals and
std_resid is the standardized training residual pool, matching the
shared ForecastResult / BHS-VaR interface used by every other model.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from tsvar import SEED
from tsvar.data import make_supervised
from tsvar.var import ForecastResult


def _model(kind):
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    if kind == "xgb":
        return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             random_state=SEED, n_jobs=-1)
    raise ValueError(kind)


def walk_forward_ml(train, test, model_kind, n_lags=5) -> ForecastResult:
    Xtr, ytr = make_supervised(train.values, n_lags)
    model = _model(model_kind).fit(Xtr, ytr)
    resid = ytr - model.predict(Xtr)
    sigma = float(np.std(resid))
    std_resid = (resid - resid.mean()) / sigma
    hist = list(train.values[-n_lags:])
    mu = np.empty(len(test))
    for t in range(len(test)):
        mu[t] = float(model.predict(np.array(hist[-n_lags:]).reshape(1, -1))[0])
        hist.append(test.values[t])  # feed realized return
    return ForecastResult(mu=mu, sigma=np.full(len(test), sigma), std_resid=std_resid,
                          y_true=test.values, dates=test.index,
                          name=getattr(test, "name", None) or "series")
