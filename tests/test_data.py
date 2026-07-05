import numpy as np, pandas as pd, pytest
from pathlib import Path
from tsvar.data import (load_index, log_returns, load_returns, train_test_returns,
                         make_supervised)

DATA = Path("data (1)/data")

def test_load_index_is_chronological_and_numeric():
    df = load_index(DATA / "ADI.csv")
    assert df.index.is_monotonic_increasing            # oldest first
    assert df["Price"].dtype == float
    assert df["Price"].iloc[0] > 0
    assert df.index[0].year == 2005                     # earliest row

def test_volume_dash_becomes_nan():
    df = load_index(DATA / "MASI.csv")                  # MASI volume is "-"
    assert df["Volume"].isna().all()

def test_log_returns_match_manual():
    p = pd.Series([100.0, 101.0, 99.0])
    r = log_returns(p)
    assert len(r) == 2
    assert r.iloc[0] == pytest.approx(100*np.log(101/100))

def test_train_test_split_shapes():
    tr, te = train_test_returns("ADI", DATA)
    assert len(tr) > len(te) > 100
    assert tr.index.max() < te.index.min()

def test_make_supervised_shapes():
    returns = np.arange(20, dtype=float)
    n_lags = 5
    X, y = make_supervised(returns, n_lags)
    assert X.shape == (len(returns) - n_lags, n_lags)
    assert y.shape == (len(returns) - n_lags,)
