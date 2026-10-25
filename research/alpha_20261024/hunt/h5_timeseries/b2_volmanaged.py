"""Family B2: VOLATILITY-MANAGED time-series exposure (1d + 4h).  h5_timeseries, 2026-10-24.

Hypothesis (Moreira & Muir 2017; Barroso & Santa-Clara 2015 analogue in crypto): realised vol is
persistent but does not forecast returns proportionally, so scaling exposure by 1/sigma_hat raises
the Sharpe of a long-crypto exposure and cuts the left tail (crashes cluster in high-vol regimes).
Who is on the other side: leveraged longs/liquidation cascades concentrated in high-vol states,
constant-notional holders. Risk: in crypto, high vol is often UP-vol (2021 alt season), so vol
scaling can also cut the best days.

PRE-REGISTRATION (written before running)
-----------------------------------------
Assets: BTCUSDT and EW20 (equal-weight top-20 PIT, rebalanced daily; 4h version uses the daily
universe mapped to 4h without lookahead).
Exposure e_t = min(cap, 0.50 / sigma_hat_t), sigma_hat = sqrt(bars/yr)*stdev of the asset's own bar
returns over the trailing L days (inclusive of bar t). Band: change exposure only if
|e_target - e_held| > 0.10. W = e_t * base weights. Long-or-flat (never short).
  G1 1d : asset {BTC, EW20} x L {7,14,30,60}d x cap {1,2}                        = 16
  G2 4h : asset {BTC, EW20} x L {3,7,14}d (4h bars) x cap 1                       = 6
  G3 1d drawdown control on G1(L=30,cap=1): e_t=0 while the asset index is more than X below its
          trailing 180d max ; X {20%,30%,40%} x asset 2                          = 6
Total 28 configs. Judged on: Sharpe vs same-asset buy&hold, alpha t-stat of managed on unmanaged
(daily OLS with HAC), sub-period Sharpe, 2x cost, lag 1.
"""
import numpy as np, pandas as pd
from common import *

ctrl = lab.controls()
U20 = daily_universe(20)
NTR = 28


def banded(target, band=0.10):
    t = target.values
    out = np.zeros_like(t)
    cur = 0.0
    for i, x in enumerate(t):
        if np.isnan(x):
            out[i] = cur
            continue
        if abs(x - cur) > band:
            cur = x
        out[i] = cur
    return pd.Series(out, index=target.index)


def alpha_reg(dm, du):
    j = pd.concat([dm.rename("m"), du.rename("u")], axis=1).dropna()
    b, t, n = hac_t(j["m"].values, j["u"].values[:, None])
    return b[0] * 365, t[0], b[1]


rows = []


def run(tf, asset, L, cap, dd=None):
    bpd = {"1d": 1, "4h": 6}[tf]
    C = lab.load("close", tf)
    if asset == "BTC":
        base = pd.DataFrame({"BTCUSDT": 1.0}, index=C.index)
        r = lab.returns(tf)["BTCUSDT"]
    else:
        U = U20 if tf == "1d" else map_daily_to_intraday(U20, C.index, tf).fillna(False)
        base = ew_weights(U)
        r = basket_returns(U, tf)
    sig = r.rolling(L * bpd, min_periods=int(L * bpd * 0.8)).std() * np.sqrt(365 * bpd)
    tgt = (0.50 / sig).clip(upper=cap)
    if dd is not None:
        idx = (1 + r.fillna(0)).cumprod()
        ddn = idx / idx.rolling(180 * bpd, min_periods=1).max() - 1
        tgt = tgt.where(ddn > -dd, 0.0)
    e = banded(tgt)
    W = base.mul(e, axis=0)
    name = f"{asset} {tf} L{L} cap{cap}" + (f" DD{int(dd*100)}" if dd else "")
    o, res = full_eval(W, tf, name, NTR, ctrl)
    bh = lab.backtest(base, tf)
    dm, du = daily(res), lab.to_daily(bh["df"]["r"], tf)
    du = du[du.index >= dm.index[0]]
    a, at, beta = alpha_reg(dm, du)
    o.update({"bh_sharpe": sharpe(du), "bh_sub": sub_sharpes(du), "alpha_ann": a, "alpha_t": at, "beta": beta,
              "avg_exposure": float(e[e.index >= res["df"].index[0] - TF_DELTA[tf]].mean())})
    rows.append(o)
    print(f"{name:28s} Sh {o['sharpe']:.2f} vs B&H {o['bh_sharpe']:.2f} | alpha {a:+.1%}/yr t {at:+.2f} beta {beta:.2f} "
          f"| c2x {o['sharpe_cost2x']:.2f} lag1 {o['sharpe_lag1']:.2f} | CAGR {o['cagr']:+.1%} DD {o['maxdd']:.1%} "
          f"exp {o['avg_exposure']:.2f} cost/yr {o['cost_drag_per_year']:.1%} fund/yr {o['funding_per_year']:+.1%} "
          f"| {fmt_sub(o['sub'])}", flush=True)


for asset in ["BTC", "EW20"]:
    for L in [7, 14, 30, 60]:
        for cap in [1, 2]:
            run("1d", asset, L, cap)
for asset in ["BTC", "EW20"]:
    for dd in [0.20, 0.30, 0.40]:
        run("1d", asset, 30, 1, dd)
for asset in ["BTC", "EW20"]:
    for L in [3, 7, 14]:
        run("4h", asset, L, 1)

df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("sub", "bh_sub")} |
                   {f"sub_{k}": v for k, v in r["sub"].items()} for r in rows])
df.to_csv("b2_volmanaged_results.csv", index=False)
print("\nB&H sub-period Sharpes:")
for r in rows:
    if "L30 cap1" in r["name"] and "DD" not in r["name"]:
        print(r["name"], fmt_sub(r["bh_sub"]))
print("configs counted:", len(rows))
