"""Final metric sheet for the B1R candidates (no new configs). h5_timeseries."""
import json
import numpy as np, pandas as pd
from common import *

ctrl = lab.controls()
Ud = daily_universe(20)
tf = "4h"; n30 = 180
C = lab.load("close", tf); R = lab.returns(tf)
U = map_daily_to_intraday(Ud, C.index, tf).fillna(False).astype(bool)
U.loc[:, "BTCUSDT"] = False
btc = R["BTCUSDT"]
z = btc / btc.rolling(n30, min_periods=n30 // 2).std()
W_alt = ew_weights(U)
out = {}
for (trig, k, h) in [("down", 2.5, 1), ("down", 1.5, 6)]:
    ev = (z < -k)
    pos = ev.astype(float).replace(0, np.nan).ffill(limit=h - 1).fillna(0.0) if h > 1 else ev.astype(float)
    W = W_alt.mul(pos, axis=0)
    o, res = full_eval(W, tf, f"4h shockrev {trig} k{k} h{h}", 46, ctrl)
    df = res["df"]; held = df["gross_exposure"] > 0
    m = res["metrics"]
    o["is_t_nw"] = m["t_nw"]; o["skew"] = m["skew"]; o["kurt"] = m["kurt"]
    o["avg_n_pos_when_in"] = float(df.loc[held, "n_pos"].mean())
    o["min_equity_usd_1x"] = float(10 * df.loc[held, "n_pos"].max())
    o["price_sharpe"] = sharpe(lab.to_daily(df["gross_r"], tf))
    o["price_ret_per_year"] = float(df["gross_r"].mean() * 2190)
    o["fund_ret_per_year"] = float(df["fund_r"].mean() * 2190)
    o["cost_per_year"] = float(df["cost_r"].mean() * 2190)
    o["n_events"] = int(ev.sum())
    # equity path from 200 USD
    d = daily(res)
    eq = 200 * (1 + d).cumprod()
    o["eq_200_end"] = float(eq.iloc[-1]); o["eq_200_min"] = float(eq.min())
    print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in o.items()}, indent=1, default=str))
    out[o["name"]] = o
pd.DataFrame([{k: v for k, v in r.items() if k != "sub"} | {f"sub_{k}": v for k, v in r["sub"].items()}
              for r in out.values()]).to_csv("final_eval.csv", index=False)
