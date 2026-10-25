import sys; sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab, numpy as np, pandas as pd
C = lab.load("close", "hype_1d")
fc = C.notna().idxmax()[C.notna().any()].sort_values()
print(fc.head(15).to_string())
print(list(C.columns[:10]))
U = lab.hype_universe("1d", top_k=50); n = U.sum(axis=1)
print(n.resample("ME").min().loc["2020"].to_string())
