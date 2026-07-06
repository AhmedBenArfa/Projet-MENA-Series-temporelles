import numpy as np, pandas as pd, pytest
from tsvar.ml import walk_forward_ml
from tsvar.var import ForecastResult

def _ret(n,seed):
    rng=np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n),
                     index=pd.date_range("2010-01-01",periods=n,freq="B"))

@pytest.mark.parametrize("kind",["rf","xgb"])
def test_ml_walk_forward(kind):
    tr,te=_ret(300,1),_ret(40,2)
    fc=walk_forward_ml(tr,te,kind,n_lags=5)
    assert isinstance(fc,ForecastResult)
    assert len(fc.mu)==len(te) and np.all(fc.sigma>0)
