"""Family A (part 1, 1d): DAY-OF-WEEK seasonality.  h5_timeseries, 2026-10-24.

PRE-REGISTRATION (written before running)
-----------------------------------------
Assets: BTCUSDT, ETHUSDT, EW20 = equal-weight top-20 perps by trailing-30d median quote volume
(point-in-time lab.universe("1d", top_k=20, min_hist_days=60)), rebalanced daily.
Estimation 2020-01-01..2022-12-31, validation 2023-01-01..2024-12-31 (both inside IS).
Descriptive: mean daily return + t by weekday (UTC day), estimation vs validation.

Trading configs (long-or-flat, 0.07%/side + funding), 3 rules x 3 assets = 9 configs:
  R1 weekend_long  : hold long only Sat+Sun UTC bars (enter Fri 24:00, exit Sun 24:00)
                     story: retail-driven weekend flow / thin books -> weekend drift?
  R2 weekday_long  : hold long only Mon..Fri bars (flat over weekend)
                     story: weekend gap risk without TradFi hedgers -> weekend carries no premium
  R3 best2_est     : long the 2 weekdays with the highest ESTIMATION-period mean; judged on validation
Benchmark: buy&hold of same asset (Sharpe), since a long/flat timing rule must beat B&H risk-adjusted.
Count: 9 configs.
"""
import numpy as np, pandas as pd
from common import *

c = lab.load("close", "1d")
R = lab.returns("1d")
U20 = daily_universe(20)
ew = basket_returns(U20, "1d")
series = {"BTC": R["BTCUSDT"], "ETH": R["ETHUSDT"], "EW20": ew}
DN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

print("=== Descriptive: mean daily return (bps) and t by weekday ===")
desc = {}
for nm, s in series.items():
    s = s.dropna()
    rows = []
    for per, (a, b) in {"EST": EST, "VAL": VAL}.items():
        x = s[(s.index >= ts(a)) & (s.index < ts(b))]
        g = x.groupby(x.index.dayofweek)
        mu = g.mean() * 1e4
        t = g.mean() / (g.std() / np.sqrt(g.count()))
        rows.append(pd.DataFrame({f"{per}_bps": mu.round(1), f"{per}_t": t.round(2)}))
    tab = pd.concat(rows, axis=1)
    tab.index = DN
    desc[nm] = tab
    print(f"\n{nm}")
    print(tab.to_string())
    # weekend vs weekday difference
    for per, (a, b) in {"EST": EST, "VAL": VAL}.items():
        x = s[(s.index >= ts(a)) & (s.index < ts(b))]
        we = x[x.index.dayofweek >= 5]; wd = x[x.index.dayofweek < 5]
        diff = we.mean() - wd.mean()
        se = np.sqrt(we.var() / len(we) + wd.var() / len(wd))
        print(f"  {per}: weekend-weekday mean diff {diff*1e4:+.1f} bps (t {diff/se:+.2f}); "
              f"weekend vol {we.std()*1e2:.2f}% vs weekday {wd.std()*1e2:.2f}%")
    # rank correlation of weekday means est vs val (persistence)
    print(f"  Spearman(EST means, VAL means) = {tab['EST_bps'].corr(tab['VAL_bps'], method='spearman'):.2f}")

ctrl = lab.controls()
results = []
NTR = 9
for nm in ["BTC", "ETH", "EW20"]:
    if nm == "EW20":
        base = ew_weights(U20)
    else:
        base = pd.DataFrame({f"{nm}USDT": 1.0}, index=c.index)
    s = series[nm].dropna()
    xe = s[(s.index >= ts(EST[0])) & (s.index < ts(EST[1]))]
    best2 = list(xe.groupby(xe.index.dayofweek).mean().nlargest(2).index)
    # held day = dayofweek of bar t+1 ; W row t (bar open t) earns bar t+1
    nxt = pd.Series((c.index + pd.Timedelta("1D")).dayofweek, index=c.index)
    rules = {"weekend_long": nxt >= 5, "weekday_long": nxt < 5, f"best2_est{[DN[i] for i in best2]}": nxt.isin(best2)}
    bh, _ = full_eval(base, "1d", f"{nm} buy&hold", 1, ctrl)
    print(f"\n{nm} B&H: Sharpe {bh['sharpe']:.2f} sub {fmt_sub(bh['sub'])}")
    results.append(bh)
    for rn, mask in rules.items():
        W = base.mul(mask.astype(float), axis=0)
        o, res = full_eval(W, "1d", f"{nm} {rn}", NTR, ctrl)
        d = daily(res)
        o["est_sharpe"] = sharpe(d[(d.index >= ts(EST[0])) & (d.index < ts(EST[1]))])
        o["val_sharpe"] = sharpe(d[(d.index >= ts(VAL[0])) & (d.index < ts(VAL[1]))])
        results.append(o)
        print(f"{o['name']:38s} Sh {o['sharpe']:.2f} (gross {o['gross_sharpe']:.2f}) EST {o['est_sharpe']:.2f} VAL {o['val_sharpe']:.2f} "
              f"cost2x {o['sharpe_cost2x']:.2f} lag1 {o['sharpe_lag1']:.2f} CAGR {o['cagr']:+.1%} DD {o['maxdd']:.1%} "
              f"cost/yr {o['cost_drag_per_year']:.1%} corrBTC {o['corr_btc']:.2f} | {fmt_sub(o['sub'])}")

pd.DataFrame([{k: v for k, v in r.items() if k != "sub"} | {f"sub_{k}": v for k, v in r["sub"].items()}
              for r in results]).to_csv("a1_dow_results.csv", index=False)
print("\nconfigs counted:", NTR)
