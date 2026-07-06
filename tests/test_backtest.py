import numpy as np, pytest
from tsvar.backtest import violations, kupiec_pof, christoffersen, backtest_summary

def test_violations_flags_breaches():
    y   = np.array([-1., -3., 0.5, -2.])
    var = np.array([-2., -2., -2., -2.])
    assert violations(y,var).tolist() == [False, True, False, False]

def test_kupiec_not_rejected_at_correct_rate():
    n=1000; alpha=0.05
    res = kupiec_pof(n_viol=50, n_obs=n, alpha=alpha)   # exactly expected
    assert res["reject"] is False and res["pvalue"] > 0.9

def test_kupiec_rejected_when_far_off():
    res = kupiec_pof(n_viol=200, n_obs=1000, alpha=0.05)  # 20% >> 5%
    assert res["reject"] is True and res["pvalue"] < 0.01

def test_kupiec_zero_violations_is_rejected():
    res = kupiec_pof(n_viol=0, n_obs=1000, alpha=0.05)
    assert res["LR"] > 50 and res["reject"] is True and res["pvalue"] < 0.01

def test_kupiec_all_violations_is_rejected():
    res = kupiec_pof(n_viol=1000, n_obs=1000, alpha=0.05)
    assert res["reject"] is True

def test_summary_keys():
    rng=np.random.default_rng(0)
    y=rng.standard_normal(250); var=np.full(250,-1.645)
    s=backtest_summary(y,var,0.05)
    for k in ["expected_rate","observed_rate","kupiec","christoffersen","basel_zone"]:
        assert k in s
