"""Family A (part 2, 1h): HOUR-OF-DAY / SESSION seasonality.  h5_timeseries, 2026-10-24.

Stories: (i) US cash-equity open (13:30-16:00 UTC) brings ETF/TradFi risk flows; (ii) Asia
session retail flow; (iii) Binance funding settles 00/08/16 UTC: longs paying high funding close
just before settlement and re-open after -> price pressure in the hour ending at settlement.

PRE-REGISTRATION (written before running)
-----------------------------------------
Assets: BTCUSDT, ETHUSDT, EW20 (daily top-20 PIT universe mapped to hourly without lookahead).
Estimation 2020-2022, validation 2023-2024 (both IS; validation never used to pick anything).
Descriptive: mean return (bps) and t by UTC bar-open hour; correlation of the 24-hour profile
estimation vs validation.
Trading configs (position held only inside the window, flat otherwise; 0.07%/side + funding):
  direction for W1..W7 = sign of the ESTIMATION-period mean return of that window (per asset)
  W1 Asia     bars 00..07     W2 Europe  bars 08..12     W3 US-open bars 13..15
  W4 US       bars 13..20     W5 pre-funding bars {07,15,23} (hour ending at settlement)
  W6 post-funding bars {00,08,16}
  W7 top3     the 3 hours with highest |t| in estimation, each with its own sign
  W8 funding-conditional pre-settlement (direction fixed by story, no estimation):
     in the hour ending at a settlement, hold -sign(f8) per coin when |f8| >= 0.03%, where
     f8 = funding settled over the trailing 8 bars (last settled 8h rate, strictly known).
8 rules x 3 assets = 24 configs.
"""
import numpy as np, pandas as pd
from common import *

C = lab.load("close", "1h")
R = lab.returns("1h")
U20 = map_daily_to_intraday(daily_universe(20), C.index, "1h").fillna(False).astype(bool)
ew = basket_returns(U20, "1h")
series = {"BTC": R["BTCUSDT"], "ETH": R["ETHUSDT"], "EW20": ew}
hr = pd.Series(C.index.hour, index=C.index)
nxt_hr = (hr + 1) % 24          # hour of bar t+1 = bar earned by W[t]

print("=== Descriptive: mean hourly return (bps) / t by bar-open hour UTC ===")
prof = {}
for nm, s in series.items():
    out = {}
    for per, (a, b) in {"EST": EST, "VAL": VAL}.items():
        x = s[(s.index >= ts(a)) & (s.index < ts(b))].dropna()
        g = x.groupby(x.index.hour)
        out[f"{per}_bps"] = (g.mean() * 1e4).round(2)
        out[f"{per}_t"] = (g.mean() / (g.std() / np.sqrt(g.count()))).round(2)
    tab = pd.DataFrame(out)
    prof[nm] = tab
    print(f"\n{nm}  corr(EST,VAL profile)={tab['EST_bps'].corr(tab['VAL_bps']):.2f}  "
          f"spearman={tab['EST_bps'].corr(tab['VAL_bps'], method='spearman'):.2f}")
    print(tab.T.to_string())
pd.concat(prof, axis=1).to_csv("a2_hour_profile.csv")

WIN = {"W1_asia": list(range(0, 8)), "W2_europe": list(range(8, 13)), "W3_usopen": [13, 14, 15],
       "W4_us": list(range(13, 21)), "W5_prefund": [7, 15, 23], "W6_postfund": [0, 8, 16]}
ctrl = lab.controls()
NTR = 24
rows = []
F = lab.load("fund", "1h")
f8 = F.rolling(8, min_periods=1).sum()

for nm in ["BTC", "ETH", "EW20"]:
    if nm == "EW20":
        base = ew_weights(U20)
    else:
        base = pd.DataFrame({f"{nm}USDT": 1.0}, index=C.index)
    s = series[nm]
    xe = s[(s.index >= ts(EST[0])) & (s.index < ts(EST[1]))].dropna()
    g = xe.groupby(xe.index.hour)
    tstat = g.mean() / (g.std() / np.sqrt(g.count()))
    top3 = list(tstat.abs().nlargest(3).index)
    rules = {}
    for wn, hrs in WIN.items():
        # estimation-period mean of the whole window per day (sum of hourly returns)
        sgn = np.sign(xe[xe.index.hour.isin(hrs)].mean())
        rules[f"{wn}({'L' if sgn > 0 else 'S'})"] = nxt_hr.isin(hrs).astype(float) * sgn
    sig = pd.Series(0.0, index=C.index)
    for h in top3:
        sig[nxt_hr == h] = np.sign(tstat[h])
    rules[f"W7_top3{top3}"] = sig
    for rn, mult in rules.items():
        W = base.mul(mult, axis=0)
        o, res = full_eval(W, "1h", f"{nm} {rn}", NTR, ctrl)
        d = daily(res)
        o["est_sharpe"] = sharpe(d[(d.index >= ts(EST[0])) & (d.index < ts(EST[1]))])
        o["val_sharpe"] = sharpe(d[(d.index >= ts(VAL[0])) & (d.index < ts(VAL[1]))])
        nb = (res["df"]["gross_exposure"] > 0).sum()
        o["gross_bps_per_bar"] = float(res["df"]["gross_r"].sum() / max(nb, 1) * 1e4)
        rows.append(o)
        print(f"{o['name']:40s} Sh {o['sharpe']:+.2f} gross {o['gross_sharpe']:+.2f} EST {o['est_sharpe']:+.2f} VAL {o['val_sharpe']:+.2f} "
              f"| gross {o['gross_bps_per_bar']:+.2f}bps/held-bar | c2x {o['sharpe_cost2x']:+.2f} lag1 {o['sharpe_lag1']:+.2f} "
              f"| CAGR {o['cagr']:+.1%} cost/yr {o['cost_drag_per_year']:.1%} fund/yr {o['funding_per_year']:+.1%}", flush=True)
    # W8 funding-conditional pre-settlement, per coin of the asset
    cols = list(base.columns)
    mask_hr = np.broadcast_to(nxt_hr.isin([7, 15, 23]).values[:, None], (len(C), len(cols)))
    sgn = -np.sign(f8[cols]) * (f8[cols].abs() >= 0.0003)
    if nm == "EW20":
        memb = U20[cols].astype(float)
        sgn = sgn * memb
        W = sgn.where(mask_hr, 0.0)
        n = W.abs().sum(axis=1).replace(0, np.nan)
        W = W.div(n, axis=0).fillna(0.0)   # 1x gross across active coins
    else:
        W = sgn.where(mask_hr, 0.0).fillna(0.0)
    o, res = full_eval(W, "1h", f"{nm} W8_fundcond_pre", NTR, ctrl)
    d = daily(res)
    o["est_sharpe"] = sharpe(d[(d.index >= ts(EST[0])) & (d.index < ts(EST[1]))])
    o["val_sharpe"] = sharpe(d[(d.index >= ts(VAL[0])) & (d.index < ts(VAL[1]))])
    df = res["df"]; held = df["gross_exposure"] > 0
    o["n_trades"] = int(held.sum())
    o["price_bps_per_trade"] = float(df.loc[held, "gross_r"].mean() * 1e4)
    o["fund_bps_per_trade"] = float(df.loc[held, "fund_r"].mean() * 1e4)
    rows.append(o)
    print(f"{o['name']:40s} Sh {o['sharpe']:+.2f} gross {o['gross_sharpe']:+.2f} EST {o['est_sharpe']:+.2f} VAL {o['val_sharpe']:+.2f} "
          f"| trades {o['n_trades']} price {o['price_bps_per_trade']:+.2f}bps fund {o['fund_bps_per_trade']:+.2f}bps vs cost 14bps "
          f"| c2x {o['sharpe_cost2x']:+.2f} lag1 {o['sharpe_lag1']:+.2f} CAGR {o['cagr']:+.1%}", flush=True)

pd.DataFrame([{k: v for k, v in r.items() if k != "sub"} | {f"sub_{k}": v for k, v in r["sub"].items()}
              for r in rows]).to_csv("a2_hourly_results.csv", index=False)
print("configs counted:", len(rows))
