import numpy as np, pandas as pd
from tsvar.volatility import walk_forward_garch
from tsvar.var import ForecastResult, var_series

def _ret(n,seed):
    rng=np.random.default_rng(seed)
    idx=pd.date_range("2010-01-01",periods=n,freq="B")
    return pd.Series(rng.standard_normal(n)*rng.uniform(0.5,2.0,n), index=idx)

def test_garch_gives_dynamic_sigma():
    tr,te=_ret(400,1),_ret(40,2)
    fc=walk_forward_garch(tr,te)
    assert isinstance(fc,ForecastResult) and len(fc.sigma)==len(te)
    assert fc.sigma.std() > 0                     # sigma varies day to day
    assert len(var_series(fc,0.01))==len(te)
