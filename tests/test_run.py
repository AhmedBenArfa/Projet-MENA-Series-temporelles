from pathlib import Path
import pandas as pd
from tsvar.run import run_index
DATA=Path("data (1)/data")

def test_run_index_arima_rf_small():
    df=run_index("Tunindex", DATA, models=["ARIMA","RF"], alphas=(0.05,))
    assert set(["model","alpha","MAE","observed_rate","kupiec_p","basel_zone"]).issubset(df.columns)
    assert len(df)==2                      # 2 models x 1 alpha
    assert df["MAE"].notna().all()
