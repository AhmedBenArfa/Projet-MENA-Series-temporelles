import numpy as np, pandas as pd, pytest
from tsvar.deep import walk_forward_dl
from tsvar.var import ForecastResult

def _ret(n,seed):
    rng=np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n),
                     index=pd.date_range("2010-01-01",periods=n,freq="B"))

@pytest.mark.parametrize("kind",["ann","lstm"])
def test_dl_walk_forward(kind):
    tr,te=_ret(200,1),_ret(20,2)
    fc=walk_forward_dl(tr,te,kind,window=10,epochs=3)
    assert isinstance(fc,ForecastResult)
    assert len(fc.mu)==len(te) and np.all(fc.sigma>0)
