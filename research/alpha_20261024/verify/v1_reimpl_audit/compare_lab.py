"""Step 1b: compare own re-implementation with lab.backtest / lab.universe / quantile_ls_weights."""
import sys, os, json
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab, numpy as np, pandas as pd
from reimpl import build_universe, f14_signal, target_weights, run_account, stats, fmt, avg_rank_weights
OUT = os.path.dirname(os.path.abspath(__file__))
C = lab.load("close", "1d"); Q = lab.load("qv", "1d"); F = lab.load("fund", "1d")
UL = lab.universe("1d", min_hist_days=60, top_k=50)
SL = F.rolling(14).sum()
WL = lab.quantile_ls_weights(-SL, UL, q=1/3)
Uo = build_universe(C, Q); So = f14_signal(F); Wo = target_weights(So, Uo)
def lab_stats(res, start=None):
    r = res["df"]["r"]
    if start: r = r[r.index >= start]
    return stats(r)
out = {}
cases = {
  "LAB: lab.universe + quantile_ls(q=1/3) + lab.backtest": lab.backtest(WL, "1d"),
  "LAB 2x cost": lab.backtest(WL, "1d", cost=0.0014),
  "LAB lag1": lab.backtest(WL, "1d", extra_lag=1),
  "LAB backtest on OWN weights": lab.backtest(Wo, "1d"),
}
for k, res in cases.items():
    s = lab_stats(res); out[k] = s; print(fmt(k, s))
    s2 = lab_stats(res, "2020-03-01"); print(fmt("   from 2020-03-01", s2))
    s3 = lab_stats(res, "2020-07-01"); out[k + " from 2020-07"] = s3; print(fmt("   from 2020-07-01 (>=12 names)", s3))
own = run_account(Wo, C, F); ownL = run_account(WL, C, F)
for k, df in {"OWN loop on OWN weights": own, "OWN loop on LAB weights": ownL}.items():
    s = stats(df["r"]); out[k] = s; print(fmt(k, s))
    s3 = stats(df["r"][df.index >= "2020-07-01"]); out[k + " from 2020-07"] = s3; print(fmt("   from 2020-07-01", s3))
# weight agreement
A = (np.sign(WL) == np.sign(Wo.reindex_like(WL))).where((WL != 0) | (Wo != 0))
print("share of (t,i) with nonzero weight where sign agrees lab vs own:", float(A.stack().mean()))
d = (WL - Wo).abs().sum(axis=1); print("mean L1 weight diff per day:", float(d[d.index >= "2020-07-01"].mean()))
# daily return diff lab vs own loop on the same lab weights -> pure accounting
rl = cases["LAB: lab.backtest(WL)" if False else "LAB: lab.universe + quantile_ls(q=1/3) + lab.backtest"]["df"]["r"]
j = pd.concat([rl.rename("lab"), ownL["r"].rename("own")], axis=1).dropna()
print("accounting-only corr:", j.corr().iloc[0, 1], " mean diff/yr:", (j.lab - j.own).mean() * 365)
dd = (j.lab - j.own).abs().sort_values(ascending=False).head(8); print(dd)
# early-2020 concentration
nl = (WL != 0).sum(axis=1); print("n positions LAB weights, 2020-03..06 monthly mean:", nl["2020-03":"2020-06"].groupby(nl["2020-03":"2020-06"].index.month).mean().round(1).to_dict())
print("LAB r in 2020-03 (cum):", float((1 + rl["2020-03-01":"2020-03-31"]).prod() - 1), " 2020-03..06:", float((1 + rl["2020-03-01":"2020-06-30"]).prod() - 1))
json.dump(out, open(os.path.join(OUT, "compare_lab.json"), "w"), indent=1)
