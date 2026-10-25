"""disp_impl step 1 - independent reimplementation of DISP7 from rules text only.

Rules (task text): each 00:00 UTC close t, U = lab.hype_universe("1d", top_k=50);
disp7 = XS std of 7d log returns across U; p = percentile of disp7 within trailing 365d;
state +1 (long EW alts ex-BTC +0.5, short BTC -0.5) if p > 2/3; -1 (short alts -0.5, long BTC +0.5)
if p < 1/3; else keep previous state.

Own return/cost/funding loop (W[t] traded at close t, earns C[t+1]/C[t]-1, pays fund[t+1]),
cross-checked vs lab.backtest. Implementation choices that the text leaves open are varied:
  minp   = min periods for the 365d percentile window: 365 (strict), 180, 90
  ddof   = 1
Outputs -> s1_out.txt / s1_daily.csv
"""
import sys, os, json
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import numpy as np, pandas as pd
import lab

OUT = os.path.dirname(os.path.abspath(__file__))
log = open(os.path.join(OUT, "s1_out.txt"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.write(s + "\n"); log.flush()


C = lab.load("close", "1d")
F = lab.load("fund", "1d")
BTC = "BTCUSDT"


def signal(U, minp=365, lookback=7, win=365, hi=2/3, lo=1/3):
    lr = np.log(C / C.shift(lookback))
    x = lr.where(U)
    disp = x.std(axis=1, ddof=1)
    disp = disp.where(U.sum(axis=1) >= 5)
    p = disp.rolling(win, min_periods=minp).rank(pct=True)
    st = np.zeros(len(p)); cur = 0.0
    for i, v in enumerate(p.values):
        if np.isfinite(v):
            if v > hi: cur = 1.0
            elif v < lo: cur = -1.0
        st[i] = cur
    return disp, p, pd.Series(st, index=p.index)


def weights(U, state, alt_gross=0.5, btc_gross=0.5):
    alts = U.copy(); alts[BTC] = False
    n = alts.sum(axis=1).replace(0, np.nan)
    W = alts.astype(float).div(n, axis=0).fillna(0.0).mul(state, axis=0) * alt_gross
    W[BTC] = -state * btc_gross
    W.loc[U.sum(axis=1) == 0] = 0.0
    return W


def own_bt(W, cost=lab.HYPE_COST, lag=0):
    """own loop: identical convention to lab but written independently"""
    W = W.reindex(index=C.index, columns=C.columns).fillna(0.0)
    if lag: W = W.shift(lag).fillna(0.0)
    Rn = (C.shift(-1) / C - 1)
    W = W.where(Rn.notna(), 0.0)
    w = W.values; rn = np.nan_to_num(Rn.values); fn = np.nan_to_num(F.shift(-1).values)
    gross = (w * rn).sum(1)
    fund = -(w * fn).sum(1)
    turn = np.zeros(len(w)); prev = np.zeros(w.shape[1])
    for i in range(len(w)):
        turn[i] = np.abs(w[i] - prev).sum()
        # drift to next bar
        port = gross[i]
        prev = w[i] * (1 + rn[i]) / (1 + port)
    net = gross + fund - turn * cost
    s = pd.Series(net, index=C.index + pd.Timedelta("1D"))[:-1]
    first = np.flatnonzero(np.abs(w).sum(1) > 0)
    s = s.iloc[first[0]:] if len(first) else s
    return s, pd.Series(turn, index=C.index), pd.Series(fund, index=C.index)


def stats(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    sh = r.mean() / r.std() * np.sqrt(365)
    dd = (eq / eq.cummax() - 1).min()
    return sh, dd, eq.iloc[-1] - 1


def sub(r):
    out = {}
    for name, a, b in [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"),
                       ("2023-24", "2023-01-01", "2025-01-01")]:
        x = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]
        out[name] = x.mean() / x.std() * np.sqrt(365)
    return out


UH = lab.hype_universe("1d", top_k=50)
P("HYPE universe: BTC in universe from", UH.index[UH[BTC]].min(), " median names", UH.sum(axis=1).median())
P("first date with >=5 names:", UH.index[UH.sum(axis=1) >= 5].min())
res_all = {}
for minp in [365, 180, 90]:
    disp, p, st = signal(UH, minp=minp)
    W = weights(UH, st)
    r, turn, fund = own_bt(W)
    sh, dd, tot = stats(r)
    sw = int((st.diff().abs() > 0).sum())
    first = st.index[np.flatnonzero(st.values != 0)[0]]
    yr = lab.yearly(r)
    r2, _, _ = own_bt(W, cost=2 * lab.HYPE_COST)
    rl, _, _ = own_bt(W, lag=1)
    lb = lab.backtest(W, "1d", cost=lab.HYPE_COST)
    P(f"\n=== minp={minp} first p valid {p.first_valid_index()} first state {first} ===")
    P(f"own: Sharpe {sh:.3f} MaxDD {dd:.1%} total {tot:.1%} switches {sw}  2xcost {stats(r2)[0]:.3f}  lag1 {stats(rl)[0]:.3f}")
    P(f"lab.backtest: Sharpe {lb['metrics']['sharpe']:.3f} MaxDD {lb['metrics']['maxdd']:.1%} turn/yr {lb['metrics']['turnover_per_year']:.1f} "
      f"cost/yr {lb['metrics']['cost_drag_per_year']:.2%} fund/yr {lb['metrics']['funding_per_year']:+.2%} start {lb['metrics']['start']}")
    P("subperiods:", {k: round(v, 2) for k, v in sub(r).items()})
    P(yr.round(3).T.to_string())
    P("time in state: long-spread %.2f short-spread %.2f flat %.2f" % ((st == 1).mean(), (st == -1).mean(), (st == 0).mean()))
    res_all[minp] = r
    if minp == 365:
        pd.DataFrame({"disp": disp, "p": p, "state": st}).to_csv(os.path.join(OUT, "s1_signal_minp365.csv"))
pd.DataFrame(res_all).to_csv(os.path.join(OUT, "s1_daily.csv"))

# also full (non-HYPE) top-50 universe, minp=365, for reference
UF = lab.universe("1d", top_k=50)
disp, p, st = signal(UF, minp=365)
r, _, _ = own_bt(weights(UF, st))
P(f"\nfull Binance top50 (minp365): Sharpe {stats(r)[0]:.3f} MaxDD {stats(r)[1]:.1%} sub {({k: round(v, 2) for k, v in sub(r).items()})}")
log.close()
