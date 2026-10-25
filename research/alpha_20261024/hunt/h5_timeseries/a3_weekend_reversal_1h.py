"""Family A (part 3, 1h): WEEKEND-MOVE REVERSAL when TradFi re-opens ("CME gap").  h5_timeseries.

Added after A1/A2 came back empty, BEFORE looking at any conditional weekend->Monday statistic.
Story: CME BTC futures and US ETFs are shut Fri ~21:00 UTC -> Sun ~22:00 UTC. Weekend moves happen in
thin books without TradFi hedgers/basis traders; when CME re-opens Sunday evening, basis traders and
institutional flow arbitrage/fade the weekend move (the folk "CME gap fill"). One trade per week,
so it can clear a 0.14% round trip if the effect exists.

PRE-REGISTRATION
----------------
Fixed UTC clock (DST ignored, <=1h error): weekend return WR = close at Sun 22:00 / close at Fri 21:00 - 1.
At Sun 22:00 UTC hold -sign(WR) for 24 bars (to Mon 22:00 UTC) if |WR| > thr, else flat.
assets {BTC, ETH, EW20 basket (its own WR)} x thr {0, 1 x sd of the trailing 26 weekend WRs} = 6 configs.
Direction fixed by story (reversal). Estimation 2020-22 / validation 2023-24 reported as a check.
"""
import numpy as np, pandas as pd
from common import *

C = lab.load("close", "1h")
R = lab.returns("1h")
U20 = map_daily_to_intraday(daily_universe(20), C.index, "1h").fillna(False).astype(bool)
ew = basket_returns(U20, "1h")
ctrl = lab.controls()
idx = C.index
close_ts = idx + pd.Timedelta("1h")          # row t closes at close_ts
NTR = 6
rows = []

for nm in ["BTC", "ETH", "EW20"]:
    if nm == "EW20":
        base = ew_weights(U20)
        lvl = (1 + ew.fillna(0)).cumprod()
    else:
        base = pd.DataFrame({f"{nm}USDT": 1.0}, index=idx)
        lvl = C[f"{nm}USDT"].ffill()
    # row whose close is Fri 21:00 and Sun 22:00
    fri = (close_ts.dayofweek == 4) & (close_ts.hour == 21)
    sun = (close_ts.dayofweek == 6) & (close_ts.hour == 22)
    lf = pd.Series(np.where(fri, lvl.values, np.nan), index=idx).ffill()
    wr = pd.Series(np.where(sun, lvl.values / lf.values - 1, np.nan), index=idx)
    wr_obs = wr.dropna()
    sdw = wr_obs.rolling(26, min_periods=13).std().shift(1)    # trailing, excludes current weekend
    for thr_name in ["0", "1sd"]:
        thr = 0.0 if thr_name == "0" else sdw
        sig = -np.sign(wr_obs) * (wr_obs.abs() > thr)
        pos = pd.Series(np.nan, index=idx)
        pos[sig.index] = sig.values
        pos = pos.ffill(limit=23).fillna(0.0)       # hold 24 bars incl. the entry row
        W = base.mul(pos, axis=0)
        o, res = full_eval(W, "1h", f"{nm} wkd_rev thr{thr_name}", NTR, ctrl)
        d = daily(res)
        o["est_sharpe"] = sharpe(d[(d.index >= ts(EST[0])) & (d.index < ts(EST[1]))])
        o["val_sharpe"] = sharpe(d[(d.index >= ts(VAL[0])) & (d.index < ts(VAL[1]))])
        # per-trade stats
        df = res["df"]
        g = df["gross_r"]
        trade_id = (pos != 0).astype(int).diff().fillna(0).clip(lower=0).cumsum()
        tid = trade_id.reindex(df.index - pd.Timedelta("1h")).values
        held = (pos.reindex(df.index - pd.Timedelta("1h")).values != 0)
        tr = pd.Series(g.values[held]).groupby(tid[held]).apply(lambda x: (1 + x).prod() - 1)
        o["n_trades"] = int(len(tr)); o["mean_trade_bps"] = float(tr.mean() * 1e4)
        o["trade_t"] = float(tr.mean() / tr.std() * np.sqrt(len(tr)))
        # correlation of WR with next-24h return (sign check)
        nxt = pd.Series(np.nan, index=idx)
        nxt[wr_obs.index] = (lvl.shift(-24) / lvl - 1)[wr_obs.index].values
        o["corr_WR_next24"] = float(pd.concat([wr_obs, nxt.dropna()], axis=1).dropna().corr().iloc[0, 1])
        rows.append(o)
        print(f"{o['name']:24s} Sh {o['sharpe']:+.2f} gross {o['gross_sharpe']:+.2f} EST {o['est_sharpe']:+.2f} VAL {o['val_sharpe']:+.2f} "
              f"| trades {o['n_trades']} mean {o['mean_trade_bps']:+.1f}bps (t {o['trade_t']:+.2f}) corr(WR,next24) {o['corr_WR_next24']:+.3f} "
              f"| c2x {o['sharpe_cost2x']:+.2f} lag1 {o['sharpe_lag1']:+.2f} | {fmt_sub(o['sub'])}", flush=True)

pd.DataFrame([{k: v for k, v in r.items() if k != "sub"} | {f"sub_{k}": v for k, v in r["sub"].items()}
              for r in rows]).to_csv("a3_weekend_results.csv", index=False)
print("configs counted:", len(rows))
