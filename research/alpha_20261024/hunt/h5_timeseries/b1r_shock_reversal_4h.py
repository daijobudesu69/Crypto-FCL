"""Family B1R (POST-HOC follow-up of B1): market-level SHOCK REVERSAL at 4h.  h5_timeseries.

Why this exists: the pre-registered lead-lag test (b1_leadlag.py) found the OPPOSITE sign of the
hypothesis: r_ALT(t+1) on r_BTC(t) = -0.074 (HAC t -2.4) at 4h, -0.066 (t -3.4) at 1h, and the
directional rule's short side lost ~21-47 bps per bar after BTC drops (alts bounce). This file tests
that reversal as a tradable rule. It is DATA-MOTIVATED -> trial count for the deflated Sharpe is
B1 (28) + this grid (18) = 46.
Story: fast BTC drops are liquidation / stop cascades (liquidity-demanding flow) that overshoot;
whoever provides liquidity is paid by the bounce (Nagel 2012 short-term reversal = return to
liquidity provision). Alts are more levered -> larger bounce. OVERLAP WARNING: this is close to the
existing CLR liquidation-flush basket (4h). Treat as a control/re-discovery unless it is clearly different.

PRE-REGISTRATION (written before running this file)
---------------------------------------------------
ALT = EW top-20 PIT ex BTC (4h). z = r_BTC,t / sd_30d(r_BTC) (trailing).
trigger {down: z < -k ; abs: |z| > k} x k {1.5, 2.0, 2.5} x hold h {1, 3, 6} 4h-bars  -> long ALT, else flat.
(a new trigger during a hold restarts the h-bar clock)  = 18 configs.
"""
import numpy as np, pandas as pd
from common import *

ctrl = lab.controls()
Ud = daily_universe(20)
NTR = 46
tf = "4h"; n30 = 180
C = lab.load("close", tf)
R = lab.returns(tf)
U = map_daily_to_intraday(Ud, C.index, tf).fillna(False).astype(bool)
U.loc[:, "BTCUSDT"] = False
btc = R["BTCUSDT"]
z = btc / btc.rolling(n30, min_periods=n30 // 2).std()
W_alt = ew_weights(U)
rows = []
for trig in ["down", "abs"]:
    for k in [1.5, 2.0, 2.5]:
        for h in [1, 3, 6]:
            ev = (z < -k) if trig == "down" else (z.abs() > k)
            pos = ev.astype(float).replace(0, np.nan).ffill(limit=h - 1).fillna(0.0) if h > 1 else ev.astype(float)
            W = W_alt.mul(pos, axis=0)
            name = f"4h shockrev {trig} k{k} h{h}"
            o, res = full_eval(W, tf, name, NTR, ctrl)
            df = res["df"]; held = df["gross_exposure"] > 0
            o["pct_time_in"] = float(held.mean()); o["n_events"] = int(ev.sum())
            o["gross_bps_per_held_bar"] = float(df.loc[held, "gross_r"].mean() * 1e4)
            o["fund_bps_per_held_bar"] = float(df.loc[held, "fund_r"].mean() * 1e4)
            rows.append(o)
            print(f"{name:26s} Sh {o['sharpe']:+.2f} gross {o['gross_sharpe']:+.2f} | ev {o['n_events']} in {o['pct_time_in']:.1%} "
                  f"gross {o['gross_bps_per_held_bar']:+.1f}bps/bar fund {o['fund_bps_per_held_bar']:+.2f} | c2x {o['sharpe_cost2x']:+.2f} "
                  f"lag1 {o['sharpe_lag1']:+.2f} | CAGR {o['cagr']:+.1%} DD {o['maxdd']:.1%} corrBTC {o['corr_btc']:+.2f} "
                  f"corrXS {o['corr_xsmom']:+.2f} DSR {o['dsr']:.2f} | {fmt_sub(o['sub'])}", flush=True)

pd.DataFrame([{k: v for k, v in r.items() if k != "sub"} | {f"sub_{k}": v for k, v in r["sub"].items()}
              for r in rows]).to_csv("b1r_shockrev_results.csv", index=False)
print("configs counted:", len(rows))
