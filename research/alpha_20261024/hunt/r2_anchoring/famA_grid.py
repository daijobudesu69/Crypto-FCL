"""FAMILY A - ANCHORING / 52-WEEK-HIGH (George & Hwang 2004), OHLCV only.

Hypothesis: traders anchor on a salient reference price (the N-day high). Good news that pushes
a coin near its high is under-reacted to (people sell "at the top", resistance), bad news that
leaves it far below its high is under-reacted to as well (holders wait to "get back to even").
=> nearness to the N-day high predicts XS returns beyond plain past return. Counterparty: the
anchoring retail seller near highs / bag-holders far below highs. The academic claim is that PH
SUBSUMES JT momentum in equities; in crypto it may simply BE momentum - spanning test decides.

PRE-REGISTERED GRID (written before running; 24 configs):
  score  in {PH   = close / max(high, N)                      (George-Hwang nearness to high)
             LOW  = log(close / min(low, N))                   (distance above N-day low)
             RP   = (close - minL) / (maxH - minL)             (position in N-day range)
             PHR  = XS residual of log PH on log N-day return  (anchoring net of momentum)}
  N      in {90, 180, 365}   rolling window, min_periods = 60 (i.e. "since listing" when younger)
  rebal  in {1, 7} days
  fixed: universe lab.hype_universe("1d", top_k=50); quintile L/S q=0.2, $-neutral, gross 1;
         cost lab.HYPE_COST + funding as cost; signal at close t, trade close t.
Follow-up (also pre-registered, counted as extra trials): for the best config by net Sharpe and
the best PHR config: full stress (2x cost, lag 1, legs, spanning vs xsmom14 / mom90 / all mom),
8L/8S 16-position version, universe top_k 30 and 75, full Binance top-50.
"""
import itertools, json
import numpy as np, pandas as pd
import common as cm
import lab

c = lab.load("close", "1d")
h = lab.load("high", "1d")
l = lab.load("low", "1d")
u50 = cm.U(50)

first_close = c.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)


def scores(N):
    mx = h.rolling(N, min_periods=60).max()
    mn = l.rolling(N, min_periods=60).min()
    ph = c / mx
    low = np.log(c / mn)
    rp = (c - mn) / (mx - mn)
    # N-day log return over the same window used by the anchor (falls back to first close if younger)
    lag = c.shift(N - 1)
    hist = c.notna().cumsum()
    lag = lag.where(lag.notna(), pd.DataFrame(np.broadcast_to(first_close.values, c.shape),
                                              index=c.index, columns=c.columns).where(hist >= 60))
    ret = np.log(c / lag)
    y = np.log(ph).where(u50 & ret.notna())
    x = ret.where(y.notna())
    xd = x.sub(x.mean(axis=1), axis=0)
    yd = y.sub(y.mean(axis=1), axis=0)
    b = (xd * yd).sum(axis=1) / (xd ** 2).sum(axis=1)
    phr = yd - xd.mul(b, axis=0)
    return {"PH": ph, "LOW": low, "RP": rp, "PHR": phr}


if __name__ == "__main__":
    rows = []
    for N in (90, 180, 365):
        S = scores(N)
        for sc, rebal in itertools.product(["PH", "LOW", "RP", "PHR"], [1, 7]):
            W = lab.quantile_ls_weights(S[sc], u50, q=0.2)
            o, _ = cm.quick(W, rebal)
            o.update({"score": sc, "N": N, "rebal": rebal})
            rows.append(o)
            print(f"{sc:4s} N={N:3d} rb={rebal}  SR {o['sharpe']:+.2f}  20-21 {o['sh_2020-21']:+.2f}  "
                  f"22 {o['sh_2022']:+.2f}  23-24 {o['sh_2023-24']:+.2f}  turn {o['turn']:.0f}  "
                  f"fund {o['fund_yr']:+.1%}  npos {o['npos']:.1f}", flush=True)
    g = pd.DataFrame(rows)
    g.to_csv("famA_grid.csv", index=False)
    print(len(g), "configs")
