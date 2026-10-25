"""v3_hype_design: quick data inspection (IS only, via lab.py)."""
import sys; sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab, numpy as np, pandas as pd
hs = lab.hype_symbols(); print("hype symbols", len(hs))
C = lab.load("close", "hype_1d"); F = lab.load("fund", "hype_1d")
print(C.shape, C.index.min(), C.index.max())
fc = C.notna().idxmax()[C.notna().any()]; print("first HYPE close per coin (min/median/max):", fc.min(), fc.sort_values().iloc[len(fc)//2], fc.max(), "n", len(fc))
ff = F.notna().idxmax()[F.notna().any()]; print("first HYPE fund:", ff.min(), ff.max(), len(ff))
print("n coins with HYPE close by date:"); 
for d in ["2023-01-01","2023-03-01","2023-06-01","2023-09-01","2024-01-01","2024-06-01","2024-12-30"]:
    print(d, int(C.loc[d].notna().sum()), int(F.loc[d].notna().sum()), int((F.loc[d].fillna(0)!=0).sum()))
U = lab.hype_universe("1d", top_k=50)
n = U.sum(axis=1); print("hype_universe top50 size by year:", n.groupby(n.index.year).agg(['min','median','max']).to_string())
Fb = lab.load("fund","1d")
print("median daily fund binance vs hype 2024 (BTC):", Fb.loc["2024","BTCUSDT"].median(), F.loc["2024","BTCUSDT"].median())
# does HYPE fund contain zeros prior to listing?
print(F.loc["2023-05-01":"2023-06-15", ["BTCUSDT","ETHUSDT","SOLUSDT"]].head(50).T.iloc[:, ::7])
