"""Family B1: BTC -> ALT LEAD-LAG (1h and 4h).  h5_timeseries, 2026-10-24.

Hypothesis: BTC is the price-discovery asset (deepest books, CME/ETF/institutional flow). Alt
quotes are updated by slower market makers / retail, so part of a BTC move reaches the alt basket
only in the next bar. Who is on the other side: stale alt liquidity providers and slow retail.
Why it may NOT persist: HFT cross-asset arbs close this within seconds-minutes; at 1h/4h bars
the residual is probably tiny.

PRE-REGISTRATION (written before running)
-----------------------------------------
ALT = equal-weight top-20 PIT universe excluding BTC (daily universe mapped intraday, no lookahead).
Descriptive (HAC OLS, per sub-period): r_ALT(t+1) = a + b1 r_BTC(t) + b2 r_ALT(t);
                                       r_BTC(t+1) = a + c1 r_BTC(t) + c2 r_ALT(t).
Signals (all trailing, L = 30d window): z = r_BTC,t / sd(r_BTC);  beta = rolling beta(ALT on BTC);
  gap = beta*r_BTC,t - r_ALT,t ; zg = gap / sd(gap).
  D  directional : ALT basket weight = sign(z) if |z|>k (else 0)
  H  hedged      : long ALT / short beta*BTC, times sign(zg) if |zg|>k (alt "catch-up" only)
  grid tf {1h,4h} x form {D,H} x k {0.5,1,2} x hold {1,3 bars} (latest signal kept up to hold bars) = 24
  XS beta catch-up: w_i ~ demeaned(beta_i * r_BTC - r_i) over last L bars, top-20 ex BTC,
     $-neutral gross 1; tf {1h,4h} x L {1,3 bars}                                                = 4
Total 28 configs.
"""
import numpy as np, pandas as pd
from common import *

ctrl = lab.controls()
Ud = daily_universe(20)
ever = [c for c in Ud.columns[Ud.any()] if c != "BTCUSDT"]
NTR = 28
rows = []
reg_rows = []


def roll_beta(y, x, n):
    mxy = (y.mul(x, axis=0)).rolling(n, min_periods=n // 2).mean()
    mx = x.rolling(n, min_periods=n // 2).mean()
    my = y.rolling(n, min_periods=n // 2).mean()
    vx = x.rolling(n, min_periods=n // 2).var(ddof=0)
    if isinstance(y, pd.DataFrame):
        return (mxy - my.mul(mx, axis=0)).div(vx, axis=0)
    return (mxy - my * mx) / vx


def hold(sig, h):
    s = sig.replace(0, np.nan)
    if h > 1:
        s = s.ffill(limit=h - 1)
    return s.fillna(0.0)


for tf in ["4h", "1h"]:
    bpd = {"4h": 6, "1h": 24}[tf]
    n30 = 30 * bpd
    C = lab.load("close", tf)
    R = lab.returns(tf)
    U = map_daily_to_intraday(Ud, C.index, tf).fillna(False).astype(bool)
    U.loc[:, "BTCUSDT"] = False
    alt = basket_returns(U, tf)
    btc = R["BTCUSDT"]
    # ---------- descriptive regressions ----------
    for per, (a, b) in list(SUBP.items()) + [("ALL", ("2020-01-01", "2025-01-01"))]:
        m = (C.index >= ts(a)) & (C.index < ts(b))
        y_alt = alt.shift(-1)[m].values; y_btc = btc.shift(-1)[m].values
        X = np.column_stack([btc[m].values, alt[m].values])
        b1, t1, n = hac_t(y_alt, X)
        c1, tc, _ = hac_t(y_btc, X)
        reg_rows.append({"tf": tf, "period": per, "n": n, "ALT<-BTC b1": b1[1], "t": t1[1], "ALT<-ALT b2": b1[2], "t2": t1[2],
                         "BTC<-BTC c1": c1[1], "tc1": tc[1], "BTC<-ALT c2": c1[2], "tc2": tc[2]})
        print(f"{tf} {per:8s} n={n:6d} ALT(t+1)~ BTC(t) b1={b1[1]:+.3f} (t {t1[1]:+.2f}) ALT(t) b2={b1[2]:+.3f} (t {t1[2]:+.2f}) | "
              f"BTC(t+1)~ BTC(t) {c1[1]:+.3f} (t {tc[1]:+.2f}) ALT(t) {c1[2]:+.3f} (t {tc[2]:+.2f})", flush=True)
    # ---------- signals ----------
    sd_b = btc.rolling(n30, min_periods=n30 // 2).std()
    z = btc / sd_b
    beta = roll_beta(alt, btc, n30)
    gap = beta * btc - alt
    zg = gap / gap.rolling(n30, min_periods=n30 // 2).std()
    W_alt = ew_weights(U)
    for form in ["D", "H"]:
        for k in [0.5, 1.0, 2.0]:
            for h in [1, 3]:
                if form == "D":
                    pos = hold(np.sign(z) * (z.abs() > k), h)
                    W = W_alt.mul(pos, axis=0)
                else:
                    pos = hold(np.sign(zg) * (zg.abs() > k), h)
                    W = W_alt.mul(pos, axis=0)
                    W["BTCUSDT"] = -beta.fillna(1.0) * pos
                name = f"{tf} {form} k{k} h{h}"
                o, res = full_eval(W, tf, name, NTR, ctrl)
                df = res["df"]; held = df["gross_exposure"] > 0
                o["pct_time_in"] = float(held.mean())
                o["gross_bps_per_held_bar"] = float(df.loc[held, "gross_r"].mean() * 1e4)
                # long vs short leg (directional: long-signal bars vs short-signal bars)
                p_next = pos.reindex(df.index - TF_DELTA[tf]).values
                g = df["gross_r"].values
                o["long_side_bps"] = float(np.nanmean(np.where(p_next > 0, g, np.nan)) * 1e4)
                o["short_side_bps"] = float(np.nanmean(np.where(p_next < 0, g, np.nan)) * 1e4)
                rows.append(o)
                print(f"{name:18s} Sh {o['sharpe']:+.2f} gross {o['gross_sharpe']:+.2f} | in {o['pct_time_in']:.0%} "
                      f"gross {o['gross_bps_per_held_bar']:+.2f}bps/bar (L {o['long_side_bps']:+.2f} S {o['short_side_bps']:+.2f}) "
                      f"| c2x {o['sharpe_cost2x']:+.2f} lag1 {o['sharpe_lag1']:+.2f} | turn/yr {o['turnover_per_year']:.0f} "
                      f"cost/yr {o['cost_drag_per_year']:.0%} | {fmt_sub(o['sub'])}", flush=True)
    # ---------- XS beta catch-up ----------
    cols = ever
    Rs = R[cols]
    bi = roll_beta(Rs, btc, n30)
    for L in [1, 3]:
        rb = btc.rolling(L).sum()
        ri = Rs.rolling(L).sum()
        score = bi.mul(rb, axis=0) - ri
        W = lab.xs_demean_weights(score, U[cols])
        name = f"{tf} XS catchup L{L}"
        o, res = full_eval(W, tf, name, NTR, ctrl)
        rows.append(o)
        print(f"{name:18s} Sh {o['sharpe']:+.2f} gross {o['gross_sharpe']:+.2f} | c2x {o['sharpe_cost2x']:+.2f} lag1 {o['sharpe_lag1']:+.2f} "
              f"| turn/yr {o['turnover_per_year']:.0f} cost/yr {o['cost_drag_per_year']:.0%} | {fmt_sub(o['sub'])}", flush=True)
        del W, score, ri
    del bi, Rs, W_alt, U, C, R
    lab.clear()

pd.DataFrame(reg_rows).to_csv("b1_leadlag_regressions.csv", index=False)
pd.DataFrame([{k: v for k, v in r.items() if k != "sub"} | {f"sub_{k}": v for k, v in r["sub"].items()}
              for r in rows]).to_csv("b1_leadlag_results.csv", index=False)
print("configs counted:", len(rows))
