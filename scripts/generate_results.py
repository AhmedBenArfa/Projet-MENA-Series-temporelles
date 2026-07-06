import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from pathlib import Path
import pandas as pd
from tsvar.run import run_all, run_index, MODELS
from tsvar.data import train_test_returns
from tsvar.var import var_series
from tsvar import plots
DATA=Path("data (1)/data"); OUT=Path("outputs"); (OUT/"figures").mkdir(parents=True,exist_ok=True)

res=run_all(DATA); res.to_csv(OUT/"results.csv",index=False)
best=(res[res.alpha==0.01].sort_values("kupiec_p",ascending=False)
      .groupby("index").first().reset_index())
best.to_csv(OUT/"best_per_index.csv",index=False)
print(res.to_string()); print("\nBEST @99%:\n", best.to_string())
