import sys; sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab, pandas as pd, numpy as np
U = lab.universe("1d", top_k=50)
out = {}
for f in ["oi_usd", "top_pos_ls", "global_ls", "taker_ls_vol"]:
    X = lab.load(f, "1d")
    have = (X.notna() & U).sum(axis=1)
    out[f] = have.resample("MS").mean().round(1)
    print(f, "zeros:", int((X == 0).sum().sum()))
df = pd.DataFrame(out)
print(df["2021-10":"2023-03"].to_string())
