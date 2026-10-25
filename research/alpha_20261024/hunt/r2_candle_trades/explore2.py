"""coverage check 2 (no performance)."""
import sys
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import numpy as np, pandas as pd
import lab
T = lab.load("trades", "1d"); H = lab.load("high", "1d"); L = lab.load("low", "1d")
U = lab.hype_universe("1d", top_k=50)
print("trades NaN share in U:", float((T.isna() & U).sum().sum() / U.sum().sum()))
print("trades<=0 share in U:", float(((T <= 0) & U).sum().sum() / U.sum().sum()))
print("H<=L share in U:", float((((H - L) <= 0) & U).sum().sum() / U.sum().sum()))
hc = lab.load("close", "hype_1d")
n = hc.notna().sum(axis=1)
print("hype_1d names with data, by year mean:", n.groupby(n.index.year).mean().round(1).to_dict())
