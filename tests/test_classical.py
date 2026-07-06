import numpy as np, pandas as pd
from tsvar.classical import walk_forward_arima
from tsvar.var import ForecastResult, var_series

def _series(n, seed):
    rng=np.random.default_rng(seed)
    idx=pd.date_range("2010-01-01",periods=n,freq="B")
    return pd.Series(rng.standard_normal(n).cumsum()*0+rng.standard_normal(n), index=idx)

def test_walk_forward_arima_returns_valid_forecast():
    tr,te=_series(300,1),_series(40,2)
    fc=walk_forward_arima(tr,te,order=(1,0,0))
    assert isinstance(fc,ForecastResult)
    assert len(fc.mu)==len(te)==len(fc.y_true)
    assert np.all(fc.sigma>0) and len(fc.std_resid)>50
    v=var_series(fc,0.05)
    assert len(v)==len(te)
