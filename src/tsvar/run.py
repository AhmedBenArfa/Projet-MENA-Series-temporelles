"""Orchestration: fit every model on every MENA index, compute VaR, backtest.

`run_index` fits the requested models (default: all 7) on one index's
train/test split, computes per-day 95%/99% VaR via BHS, backtests each
against Basel/Kupiec/Christoffersen, and returns one tidy row per
(model, alpha). `run_all` concatenates this across the 4 MENA indices.
"""

import numpy as np
import pandas as pd

from tsvar.data import train_test_returns
from tsvar.var import var_series
from tsvar.backtest import backtest_summary
from tsvar.classical import walk_forward_arima, walk_forward_sarima
from tsvar.volatility import walk_forward_garch
from tsvar.ml import walk_forward_ml
from tsvar.deep import walk_forward_dl

MODELS = {
    "ARIMA":  lambda tr, te: walk_forward_arima(tr, te),
    "SARIMA": lambda tr, te: walk_forward_sarima(tr, te, m=5),
    "GARCH":  lambda tr, te: walk_forward_garch(tr, te),
    "RF":     lambda tr, te: walk_forward_ml(tr, te, "rf"),
    "XGB":    lambda tr, te: walk_forward_ml(tr, te, "xgb"),
    "ANN":    lambda tr, te: walk_forward_dl(tr, te, "ann"),
    "LSTM":   lambda tr, te: walk_forward_dl(tr, te, "lstm"),
}


def run_index(name, data_dir, models=None, alphas=(0.05, 0.01)) -> pd.DataFrame:
    tr, te = train_test_returns(name, data_dir)
    rows = []
    models = models or list(MODELS)
    cache = {}
    for mname in models:
        fc = cache.setdefault(mname, MODELS[mname](tr, te))
        mae = float(np.mean(np.abs(fc.y_true - fc.mu)))
        rmse = float(np.sqrt(np.mean((fc.y_true - fc.mu) ** 2)))
        for a in alphas:
            v = var_series(fc, a)
            s = backtest_summary(fc.y_true, v, a)
            rows.append({
                "index": name,
                "model": mname,
                "alpha": a,
                "MAE": mae,
                "RMSE": rmse,
                "observed_rate": s["observed_rate"],
                "kupiec_p": s["kupiec"]["pvalue"],
                "christoffersen_p": s["christoffersen"]["pvalue_cc"],
                "basel_zone": s["basel_zone"],
            })
    return pd.DataFrame(rows)


def run_all(data_dir, indices=("Tunindex", "ADI", "MASI", "TASI")) -> pd.DataFrame:
    return pd.concat([run_index(i, data_dir) for i in indices], ignore_index=True)
