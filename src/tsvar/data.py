from pathlib import Path
import numpy as np, pandas as pd

INDEX_FILES = {"ADI":"ADI.csv","CAC40":"CAC40.csv","MASI":"MASI.csv",
               "S&P500":"S&P500.csv","TASI":"TASI.csv","Tunindex":"Tunindex.csv"}
TEST_FILES  = {"ADI":"ADITest.csv","MASI":"MASITest.csv",
               "TASI":"TASITest.csv","Tunindex":"TunindexTest.csv"}
MENA = ["Tunindex","ADI","MASI","TASI"]
BENCHMARKS = ["CAC40","S&P500"]

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",","",regex=False), errors="coerce")

def _vol(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    mult = np.where(s.str.endswith("M"),1e6, np.where(s.str.endswith("K"),1e3,1.0))
    base = pd.to_numeric(s.str.replace("[MK]","",regex=True), errors="coerce")
    return base*mult

def load_index(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%b %d, %Y")
    for c in ["Price","Open","High","Low"]:
        df[c] = _num(df[c])
    df["Volume"] = _vol(df["Vol."])
    df["ChangePct"] = _num(df["Change %"].astype(str).str.replace("%","",regex=False))
    df = df.set_index("Date").sort_index()
    return df[["Price","Open","High","Low","Volume","ChangePct"]]

def log_returns(prices: pd.Series) -> pd.Series:
    return (100*np.log(prices/prices.shift(1))).dropna()

def load_returns(name, data_dir, test=False) -> pd.Series:
    fname = (TEST_FILES if test else INDEX_FILES)[name]
    r = log_returns(load_index(Path(data_dir)/fname)["Price"])
    r.name = name
    return r

def train_test_returns(name, data_dir):
    return load_returns(name, data_dir, False), load_returns(name, data_dir, True)

import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

def adf_test(series) -> dict:
    stat,p,*_ = adfuller(series.dropna(), autolag="AIC")
    return {"stat":stat,"pvalue":p,"stationary":p<0.05}

def kpss_test(series) -> dict:
    # KPSS's p-value lookup table is bounded; stationary financial returns
    # routinely produce a statistic below/above the table range, which
    # statsmodels flags via InterpolationWarning. This is expected here
    # (not a bug), so it is narrowly silenced rather than left to leak
    # into test/CI output.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InterpolationWarning)
        stat,p,*_ = kpss(series.dropna(), regression="c", nlags="auto")
    return {"stat":stat,"pvalue":p,"stationary":p>0.05}

def make_supervised(returns: np.ndarray, n_lags: int):
    X,y = [],[]
    for i in range(n_lags,len(returns)):
        X.append(returns[i-n_lags:i]); y.append(returns[i])
    return np.array(X), np.array(y)

def make_sequences(returns: np.ndarray, window: int):
    return make_supervised(returns, window)   # same shape; DL reshapes to (n,window,1)
