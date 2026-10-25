"""Diagnostics for B1R shock reversal (no new configs; descriptive only).
1) alpha of each B1R config vs the EW ALT basket buy&hold (daily HAC OLS)  -> is it just beta?
2) event concentration (share of gross PnL from the top-10% events), crash-month exclusion
3) 1h event study: hourly ALT basket returns in the 8 hours after the 4h trigger closes
   -> how fast is the bounce (practical execution delay)?
"""
import numpy as np, pandas as pd
from common import *

Ud = daily_universe(20)
tf = "4h"; n30 = 180
C = lab.load("close", tf); R = lab.returns(tf)
U = map_daily_to_intraday(Ud, C.index, tf).fillna(False).astype(bool)
U.loc[:, "BTCUSDT"] = False
btc = R["BTCUSDT"]
alt4 = basket_returns(U, tf)
z = btc / btc.rolling(n30, min_periods=n30 // 2).std()
W_alt = ew_weights(U)
bh = lab.backtest(W_alt, tf)
du = daily(bh)
print(f"ALT basket B&H 4h: Sharpe {bh['metrics']['sharpe']:.2f} | {fmt_sub(sub_sharpes(du))}")
print("unconditional ALT 4h mean bps:", round(alt4.mean() * 1e4, 2))
for trig in ["down", "abs"]:
    for k in [1.5, 2.0, 2.5]:
        for h in [1, 3, 6]:
            ev = (z < -k) if trig == "down" else (z.abs() > k)
            pos = ev.astype(float).replace(0, np.nan).ffill(limit=h - 1).fillna(0.0) if h > 1 else ev.astype(float)
            res = lab.backtest(W_alt.mul(pos, axis=0), tf)
            dm = daily(res)
            j = pd.concat([dm.rename("m"), du.rename("u")], axis=1).dropna()
            b, t, n = hac_t(j["m"].values, j["u"].values[:, None])
            frac = pos.mean()
            print(f"{trig:4s} k{k} h{h}: Sh {res['metrics']['sharpe']:+.2f} | random-timing Sh ~ {bh['metrics']['sharpe']*np.sqrt(frac):.2f} "
                  f"| alpha vs ALT B&H {b[0]*365:+.1%}/yr (t {t[0]:+.2f}) beta {b[1]:.2f}")

# event concentration and crash exclusion for down k2 h1 and down k1.5 h6
for (k, h) in [(2.0, 1), (1.5, 6), (2.5, 1)]:
    ev = (z < -k)
    pos = ev.astype(float).replace(0, np.nan).ffill(limit=h - 1).fillna(0.0) if h > 1 else ev.astype(float)
    res = lab.backtest(W_alt.mul(pos, axis=0), tf)
    df = res["df"]; g = df["gross_r"]; r = df["r"]
    held = df["gross_exposure"] > 0
    gt = g[held]
    top = gt.sort_values(ascending=False)
    share = top.iloc[:max(1, len(top) // 10)].sum() / gt.sum()
    d = daily(res)
    excl = d[~(((d.index.year == 2020) & (d.index.month == 3)) | ((d.index.year == 2021) & (d.index.month == 5)))]
    hit = (gt > 0).mean()
    print(f"down k{k} h{h}: held bars {held.sum()}, hit-rate {hit:.0%}, top-10% bars = {share:.0%} of gross PnL, "
          f"Sharpe ex Mar-2020 & May-2021 {sharpe(excl):+.2f}, median bar {gt.median()*1e4:+.1f}bps mean {gt.mean()*1e4:+.1f}bps")
    print("   yearly:", yearly_str(d))

# 1h event study
lab.clear()
C1 = lab.load("close", "1h")
U1 = map_daily_to_intraday(Ud, C1.index, "1h").fillna(False).astype(bool)
U1.loc[:, "BTCUSDT"] = False
alt1 = basket_returns(U1, "1h")
for k in [1.5, 2.0, 2.5]:
    ev_ts = z.index[(z < -k).values]                  # 4h bar open ts; closes at ts+4h
    t0 = ev_ts + pd.Timedelta("4h")                    # first 1h bar after the trigger close
    out = []
    for j in range(8):
        x = alt1.reindex(t0 + pd.Timedelta(hours=j)).dropna()
        out.append((j + 1, x.mean() * 1e4, x.mean() / x.std() * np.sqrt(len(x))))
    print(f"down k{k}: n={len(ev_ts)} hourly ALT return after trigger (hour, bps, t): " +
          "  ".join(f"h{a}:{b:+.1f}({c:+.1f})" for a, b, c in out))
print("unconditional ALT 1h mean bps:", round(alt1.mean() * 1e4, 2))
