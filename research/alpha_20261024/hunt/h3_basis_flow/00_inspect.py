"""Data sanity inspection (no performance computed). Checks coverage/units of fields used by h3."""
import sys
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import numpy as np, pandas as pd
import lab

for f in ["close", "prem_mean", "prem", "spot_close", "fund", "qv", "tbqv", "trades"]:
    P = lab.load(f, "1d")
    nn = P.notna().sum(axis=1)
    print(f"{f:12s} shape {P.shape} first {P.index[0].date()} last {P.index[-1].date()} "
          f"n_valid@2020-06 {nn.loc['2020-06-01']} "
          f"@2022-01 {nn.loc['2022-01-01']} @2024-06 {nn.loc['2024-06-01']}")
    s = P["BTCUSDT"].dropna()
    print("   BTC first", s.index[0].date(), "describe:", s.describe()[["mean", "std", "min", "max"]].round(6).to_dict())

pm = lab.load("prem_mean", "1d"); fd = lab.load("fund", "1d")
sc = lab.load("spot_close", "1d"); c = lab.load("close", "1d")
basis = c / sc - 1
print("BTC basis describe", basis["BTCUSDT"].describe().round(5).to_dict())
print("ETH prem_mean vs fund corr", pm["ETHUSDT"].corr(fd["ETHUSDT"]))
# pooled corr of prem_mean*3 vs fund (daily fund = sum of 3 settlements)
x = (pm * 3).stack(); y = fd.stack()
j = pd.concat([x, y], axis=1).dropna()
print("pooled corr 3*prem_mean vs fund", j.corr().iloc[0, 1], " mean 3*pm", j[0].mean(), " mean fund", j[1].mean())
x = basis.stack(); j = pd.concat([x, pm.stack()], axis=1).dropna()
print("pooled corr basis vs prem_mean", j.corr().iloc[0, 1])
tb = lab.load("tbqv", "1d"); qv = lab.load("qv", "1d")
r = (tb / qv)
print("taker ratio pooled describe", r.stack().describe().round(4).to_dict())
U = lab.universe("1d", top_k=50)
print("universe size by year", U.sum(axis=1).groupby(U.index.year).mean().round(1).to_dict())
spot_avail = (sc.notna() & U).sum(axis=1) / U.sum(axis=1)
print("frac of U with spot by year", spot_avail.groupby(spot_avail.index.year).mean().round(2).to_dict())
pm_avail = (pm.notna() & U).sum(axis=1) / U.sum(axis=1)
print("frac of U with prem_mean by year", pm_avail.groupby(pm_avail.index.year).mean().round(2).to_dict())
