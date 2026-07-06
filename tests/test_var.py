import numpy as np, pandas as pd, pytest
from tsvar.var import bhs_quantile, var_series, ForecastResult

def test_bhs_quantile_recovers_normal_var():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(200_000)          # standardized residuals ~ N(0,1)
    q = bhs_quantile(mu=0.0, sigma=1.0, std_resid=z, alpha=0.05, n_boot=200_000, seed=1)
    assert q == pytest.approx(-1.645, abs=0.05)   # 5% normal quantile

def test_bhs_location_scale_shift():
    z = np.array([-2.,-1.,0.,1.,2.])
    q0 = bhs_quantile(0.0,1.0,z,0.5,n_boot=50_000,seed=2)   # median ~ 0
    q1 = bhs_quantile(3.0,2.0,z,0.5,n_boot=50_000,seed=2)   # shifted by mu, scaled
    assert q1 == pytest.approx(3.0 + 2.0*q0, abs=0.1)

def test_var_series_length():
    n=30; rng=np.random.default_rng(3)
    fc = ForecastResult(mu=np.zeros(n), sigma=np.ones(n),
                        std_resid=rng.standard_normal(500),
                        y_true=rng.standard_normal(n),
                        dates=pd.date_range("2015-01-01",periods=n), name="X")
    v = var_series(fc, 0.05)
    assert v.shape == (n,)
    assert np.all(v < 0)                       # left-tail returns negative
