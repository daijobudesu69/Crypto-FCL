"""r2_candle_trades - data coverage check only (no performance numbers). IS data via lab.py."""
import sys
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import numpy as np, pandas as pd
import lab

C = lab.load("close", "1d"); H = lab.load("high", "1d"); L = lab.load("low", "1d"); O = lab.load("open", "1d")
T = lab.load("trades", "1d"); Q = lab.load("qv", "1d")
print("1d range", C.index[0], C.index[-1], C.shape)
for k in (30, 50, 75):
    U = lab.hype_universe("1d", top_k=k)
    n = U.sum(axis=1)
    print("hype top", k, "names per year (mean):", n.groupby(n.index.year).mean().round(1).to_dict())
U = lab.hype_universe("1d", top_k=50)
print("trades NaN share where close present & in U:", float(T.where(U).isna().sum().sum() / U.sum().sum()))
print("trades==0 share in U:", float((T.where(U) == 0).sum().sum() / U.sum().sum()))
print("H==L share in U:", float(((H - L).where(U) <= 0).sum().sum() / U.sum().sum()))
ats = (Q / T).where(U)
print("avg trade size USD median by year:", ats.median(axis=1).groupby(ats.index.year).median().round(0).to_dict())
for f in ["close", "high", "low", "open", "qv"]:
    h = lab.load(f, "hype_1d")
    print("hype_1d", f, h.index[0], h.index[-1], h.shape, "non-null cols", int(h.notna().any().sum()))
h = lab.load("close", "hype_1d")
print("hype_1d first valid by col (sample):", h.apply(lambda s: s.first_valid_index()).dropna().sort_values().head(3).to_dict())
print("hype_1d column names sample", list(h.columns[:5]))
