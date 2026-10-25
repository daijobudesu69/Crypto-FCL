"""Step 1: independent reimplementation of PKCC30 from the rules text.
PRE-REGISTERED (reproduction of hunter's claimed checks only, no new search):
  base n=30 q=0.2 gross 1.0 HYPE top50, HYPE_COST, funding on
  2x cost; extra_lag 1, 2; long-only leg, short-only leg (each at 0.5 gross, same names)
  windows 10/20/45/60/90 (minp=ceil(0.8n)); k=8 per side (16 positions) + its lag1
  cross-check of the same W through lab.backtest.
"""
import json
import numpy as np, pandas as pd
from common_pk import *

H, L, C = panels()
U = lab.hype_universe("1d", top_k=50)
out = {}
lines = []

def summ(name, df):
    r = df["r"]
    d = {"sharpe": sharpe(r), "cagr": cagr(r), "maxdd": maxdd(r), "start": str(r.index[0].date()),
         "turn_yr": float(df["turnover"].mean() * 365), "cost_yr": float(df["cost_r"].mean() * 365),
         "fund_yr": float(df["fund_r"].mean() * 365), **subp(r)}
    out[name] = d
    s = (f"{name:28s} SR {d['sharpe']:.2f} CAGR {d['cagr']:+.1%} MDD {d['maxdd']:.1%} "
         f"sub {d['2020-21']:.2f}/{d['2022']:.2f}/{d['2023-24']:.2f} turn/yr {d['turn_yr']:.0f} "
         f"cost/yr {d['cost_yr']:.1%} fund/yr {d['fund_yr']:+.1%} start {d['start']}")
    print(s); lines.append(s)
    return d

S = ratio_signal(H, L, C, 30, 24)
W = ls_weights(S, U & S.notna(), q=0.2)
base = own_bt(W)
summ("base n30", base)
print(yearly(base["r"]).round(3).T.to_string())
lines.append("yearly base:\n" + yearly(base["r"]).round(3).T.to_string())
npos = (W != 0).sum(axis=1)
nl = (W > 0).sum(axis=1); ns = (W < 0).sum(axis=1)
act = npos[npos > 0]
s = f"positions/day: mean {act.mean():.1f} min {act.min()} max {act.max()}; long mean {nl[npos>0].mean():.1f} short mean {ns[npos>0].mean():.1f}; first trade date {act.index[0].date()}"
print(s); lines.append(s)
Un = (U & S.notna()).sum(axis=1)
s = f"eligible names/day (U & ratio valid): mean {Un[Un>0].mean():.1f} min {Un[Un>0].min()} max {Un.max()} ; by year " + \
    str(Un[Un>0].groupby(Un[Un>0].index.year).mean().round(1).to_dict())
print(s); lines.append(s)

summ("2x cost", own_bt(W, cost=2 * lab.HYPE_COST))
summ("lag1", own_bt(W, extra_lag=1))
summ("lag2", own_bt(W, extra_lag=2))
summ("no funding", own_bt(W, funding=False))
WL = W.clip(lower=0); WS = W.clip(upper=0)
summ("long leg (0.5 gross)", own_bt(WL))
summ("short leg (0.5 gross)", own_bt(WS))
for n in (10, 20, 45, 60, 90):
    Sn = ratio_signal(H, L, C, n)
    summ(f"window n{n}", own_bt(ls_weights(Sn, U & Sn.notna(), q=0.2)))
W8 = ls_weights(S, U & S.notna(), k=8)
summ("k=8/side (16 pos)", own_bt(W8))
summ("k=8/side lag1", own_bt(W8, extra_lag=1))

# cross-check through lab.backtest
res = lab.backtest(W, "1d", cost=lab.HYPE_COST)
m = res["metrics"]
s = f"lab.backtest same W: SR {m['sharpe']:.2f} CAGR {m['cagr']:+.1%} MDD {m['maxdd']:.1%} start {m['start']}"
print(s); lines.append(s)
rr = res["df"]["r"].reindex(base.index)
s = f"own vs lab daily r: corr {np.corrcoef(rr.dropna(), base['r'].reindex(rr.dropna().index))[0,1]:.5f}, max abs diff {np.nanmax(np.abs(rr - base['r'])):.2e}"
print(s); lines.append(s)
print(lab.yearly(res["df"]["r"]).round(3).T.to_string())

# correlations with controls
ctl = lab.controls()
j = pd.concat([base["r"], ctl], axis=1).dropna()
s = f"corr with BTC {j.iloc[:,0].corr(j['btc']):.2f}, with 14d XS mom {j.iloc[:,0].corr(j['xsmom14']):.2f}"
print(s); lines.append(s)

W.to_parquet("W_base.parquet")
base.to_parquet("base_daily.parquet")
json.dump(out, open("s1_results.json", "w"), indent=1)
open("s1_out.txt", "w").write("\n".join(lines))
