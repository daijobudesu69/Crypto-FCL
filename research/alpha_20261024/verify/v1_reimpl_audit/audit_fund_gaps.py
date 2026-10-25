"""Audit: holes in the raw funding history (2020-2024 only) -> panel fund==0 on listed days -> F14 biased to 0."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab
RAW = r"C:\Crypto data\backtest data and more\data\binance\um\funding"
CUT = pd.Timestamp("2025-01-01", tz="UTC")
F = lab.load("fund", "1d"); C = lab.load("close", "1d")
UL = lab.universe("1d", min_hist_days=60, top_k=50)
z = (F == 0) & C.notna()
rows = []
for s in F.columns:
    nz = int(z[s].sum())
    if nz == 0: continue
    zd = z.index[z[s].values]
    # contiguous runs
    runs = []; st = zd[0]; pv = zd[0]
    for d in zd[1:]:
        if (d - pv).days > 1:
            runs.append((st, pv)); st = d
        pv = d
    runs.append((st, pv))
    inU = int((z[s] & UL[s]).sum())
    rows.append((s, nz, inU, [(a.date().isoformat(), b.date().isoformat()) for a, b in runs if (b - a).days >= 2][:6]))
for r in sorted(rows, key=lambda x: -x[1]):
    print(f"{r[0]:12s} zero-fund listed days={r[1]:4d}  of which in top-50 universe={r[2]:4d}  runs>=3d: {r[3]}")
# raw check for BNB / ICP: settlement gaps
for s in ["BNBUSDT", "ICPUSDT", "LTCUSDT", "LINKUSDT"]:
    d = pd.read_parquet(os.path.join(RAW, f"{s}.parquet"), filters=[("ts", "<", CUT)]).sort_values("ts")
    t = d["ts"]
    g = t.diff().dt.total_seconds() / 3600
    big = d.loc[g > 8.5, "ts"]
    print(s, "raw rows", len(d), "first", t.iloc[0], " gaps>8.5h:", [(str((t[t < x].iloc[-1]).date()), str(x.date()), round(float(g[t == x].iloc[0]) / 24, 1)) for x in big][:6])
# how often did zero-fund names land in the LONG leg (lowest F14) of the lab-style weights
S = F.rolling(14).sum()
W = lab.quantile_ls_weights(-S, UL, q=1/3)
z14 = (F.rolling(14).apply(lambda x: (x == 0).sum(), raw=True) >= 7)   # >= half the window is a hole
aff = (W > 0) & z14
print("coin-days in LONG leg with >=7 of 14 funding days == 0:", int(aff.values.sum()), "of", int((W > 0).values.sum()), "long coin-days")
print(aff.sum()[aff.sum() > 0].to_dict())
aff.to_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "zero_fund_long_mask.parquet"))
