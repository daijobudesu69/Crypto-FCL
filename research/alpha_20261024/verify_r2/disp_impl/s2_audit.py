"""disp_impl step 2 - exact replication (minp=120 as found in hunter code AFTER step 1) + lookahead audit.

(a) minp=120 replication, two percentile implementations (pandas rolling rank vs explicit <= count)
(b) truncation test: cut ALL loaded panels (close, qv, fund) at 3 dates, recompute universe/signal/state/
    weights/returns from scratch, compare with full-sample values strictly before the cut.
(c) misc audit counts: first valid dates, thin-universe days, forced-flat events (next-bar NaN), ties.
Outputs -> s2_out.txt
"""
import sys, os
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import numpy as np, pandas as pd
import lab

OUT = os.path.dirname(os.path.abspath(__file__))
log = open(os.path.join(OUT, "s2_out.txt"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.write(s + "\n"); log.flush()


BTC = "BTCUSDT"


def build(minp=120, pct="rank", min_names=0):
    C = lab.load("close", "1d")
    U = lab.hype_universe("1d", top_k=50)
    lr = np.log(C / C.shift(7))
    disp = lr.where(U).std(axis=1, ddof=1)
    if min_names:
        disp = disp.where(U.sum(axis=1) >= min_names)
    if pct == "rank":
        p = disp.rolling(365, min_periods=minp).rank(pct=True, method="max")
    else:
        p = disp.rolling(365, min_periods=minp).apply(
            lambda x: (x[np.isfinite(x)] <= x[-1]).mean() if np.isfinite(x[-1]) else np.nan, raw=True)
    st = np.zeros(len(p)); cur = 0.0
    for i, v in enumerate(p.values):
        if np.isfinite(v):
            if v > 2 / 3: cur = 1.0
            elif v < 1 / 3: cur = -1.0
        st[i] = cur
    st = pd.Series(st, index=p.index)
    alts = U.copy(); alts[BTC] = False
    W = alts.astype(float).div(alts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).mul(st, axis=0) * 0.5
    W[BTC] = -0.5 * st
    return U, disp, p, st, W


def sh(r):
    r = r.dropna(); return r.mean() / r.std() * np.sqrt(365)


# (a) replication
for pct in ["rank", "count"]:
    U, disp, p, st, W = build(120, pct)
    res = lab.backtest(W, "1d", cost=lab.HYPE_COST)
    r = res["df"]["r"]; m = res["metrics"]
    r2 = lab.backtest(W, "1d", cost=2 * lab.HYPE_COST)["df"]["r"]
    rl = lab.backtest(W, "1d", cost=lab.HYPE_COST, extra_lag=1)["df"]["r"]
    y = lab.yearly(r)
    sw = int((st.diff().abs() > 0).sum())
    subs = {k: round(sh(r[(r.index >= pd.Timestamp(a, tz='UTC')) & (r.index < pd.Timestamp(b, tz='UTC'))]), 2)
            for k, a, b in [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"), ("2023-24", "2023-01-01", "2025-01-01")]}
    P(f"(a) minp120 pct={pct}: Sharpe {m['sharpe']:.3f} CAGR {m['cagr']:.1%} MaxDD {m['maxdd']:.1%} tNW {m['t_nw']:.2f} start {m['start']} "
      f"switches {sw} 2x {sh(r2):.3f} lag1 {sh(rl):.3f} subs {subs} skew {m['skew']:.2f} kurt {m['kurt']:.2f} days {m['days']}")
    P("    yearly ret", {int(k): round(v, 3) for k, v in y['ret'].items()}, " SR", {int(k): round(v, 2) for k, v in y['sharpe'].items()})
    if pct == "rank":
        base = dict(U=U, disp=disp, p=p, st=st, W=W, r=r)
        for n in [95, 855]:
            P(f"    DSR n={n}: {lab.deflated_sharpe(m['sharpe'], n, m['days'], m['skew'], m['kurt'] + 3):.3f}")

# (c) misc audit
U, disp, p, st, W, r = [base[k] for k in ["U", "disp", "p", "st", "W", "r"]]
n = U.sum(axis=1)
P("\n(c) first date disp valid:", disp.first_valid_index(), " names that day", int(n.loc[disp.first_valid_index()]))
P("    first p valid:", p.first_valid_index(), " first nonzero state:", st.index[np.flatnonzero(st.values)[0]])
P("    days in-sample with state!=0 and <10 names:", int(((st != 0) & (n < 10)).sum()), " <20 names:", int(((st != 0) & (n < 20)).sum()))
P("    min / median names while trading:", int(n[st != 0].min()), int(n[st != 0].median()))
C = lab.load("close", "1d")
Rn = C.pct_change(fill_method=None).shift(-1)
forced = ((W != 0) & Rn.isna()).sum().sum()
P("    forced-flat cells (W!=0 but next-bar return NaN):", int(forced))
# NaN gaps inside a listed history
gaps = (C.isna() & C.ffill().notna() & C.bfill().notna()).sum()
P("    interior NaN closes (gaps) per coin, coins with gaps:", gaps[gaps > 0].to_dict())
P("    duplicate disp values (ties):", int(disp.dropna().duplicated().sum()))
# hunter's percentile includes today in window -> check percentile in [1/k,1]
P("    p range:", float(p.min()), float(p.max()))

# (b) truncation test
full = {f: lab.load(f, "1d").copy() for f in ["close", "qv", "fund"]}
cuts = ["2021-06-30", "2022-11-15", "2024-03-01"]
for cut in cuts:
    ct = pd.Timestamp(cut, tz="UTC")
    lab.clear()
    for f in full:
        lab._mem[(f, "1d")] = full[f][full[f].index < ct]
    Ut, dt, pt, stt, Wt = build(120, "rank")
    rt = lab.backtest(Wt, "1d", cost=lab.HYPE_COST)["df"]["r"]
    ix = dt.index
    eqU = (Ut == U.loc[ix]).all().all()
    ddisp = np.nanmax(np.abs(dt.values - disp.loc[ix].values))
    dp = np.nanmax(np.abs(pt.values - p.loc[ix].values))
    nanmis = int((pt.isna() != p.loc[ix].isna()).sum())
    est = int((stt != st.loc[ix]).sum())
    dW = float(np.abs(Wt.values - W.loc[ix, Wt.columns].values).max())
    # returns realised strictly before the cut: index (realisation date) <= cut - 1 day; last bar needs C[cut] -> excluded
    common = rt.index[rt.index < ct]
    common = common[common.isin(r.index)]
    dr = float(np.abs(rt.loc[common] - r.loc[common]).max())
    P(f"(b) cut {cut}: universe identical {eqU}, max|d disp| {ddisp:.2e}, max|d p| {dp:.2e}, p-NaN mismatches {nanmis}, "
      f"state mismatches {est}, max|dW| {dW:.2e}, returns compared {len(common)} max|dr| {dr:.2e}, last ret date {common.max()}")
lab.clear()
log.close()
