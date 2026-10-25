"""Family A3 - SESSION-BLOCK cross-sectional seasonality (lower-turnover version of A2).

Motivation (from A2 diagnostics, famA_diag.py): the 4h same-slot score has gross Sharpe ~3.4 and the
effect is 24h-periodic (24h lag keeps it, 4h/8h lags flip it), but break-even cost is ~2.6 bp/side vs
6.5 bp HYPE taker. Hypothesis: the coin-specific intraday shape is a session effect (Asia / Europe /
US hours); trading 8h or 12h blocks cuts turnover 2-3x.
Prior (written before running): lag-1 at 4h was strongly negative -> adjacent slots anti-correlate ->
block aggregation probably destroys the signal. Run anyway, small grid.

PRE-REGISTERED GRID (8 configs; family A total = 38 + 8 = 46)
  block  8h (00/08/16 UTC, 3 blocks/day)  |  12h (00/12 UTC, 2 blocks/day)
  N      30, 60 days (same block-of-day over the last N days)
  variant raw | excess (minus trailing mean block return over all blocks of the last N days)
  selection n8 (8 long / 8 short), held with drift through the block, universe hype top-50.
Cost HYPE_COST + funding; 2x cost and extra_lag=1 (4h late) reported.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm
from common import lab

N_TRIALS_A = 46


def drift_hold(Wt, R, rebal):
    """hold target weights set on rebal rows, drifting with returns until the next rebal row."""
    T = Wt.fillna(0.0).values
    Rn = R.shift(-1).reindex(index=Wt.index, columns=Wt.columns).fillna(0.0).values  # earned over t+1
    m = rebal.values
    out = np.zeros_like(T)
    w = np.zeros(T.shape[1])
    for t in range(len(T)):
        if m[t]:
            w = T[t].copy()
        out[t] = w
        gr = w * (1 + Rn[t])
        tot = 1 + (w * Rn[t]).sum()
        w = gr / tot if tot > 0 else gr
    return pd.DataFrame(out, index=Wt.index, columns=Wt.columns)


if __name__ == "__main__":
    R = lab.returns("4h").astype("float32")
    U = cm.daily_to_4h(cm.hype_U(50), R.index).reindex(index=R.index, columns=R.columns).fillna(False)
    lr = np.log1p(R)
    rows = []
    for L in [2, 3]:
        blk_h = 4 * L
        Bend = np.expm1(lr.rolling(L, min_periods=L).sum())          # block return ending at row e
        nxt_hour = (R.index + pd.Timedelta("4h")).hour
        rebal = pd.Series((nxt_hour % blk_h) == 0, index=R.index)    # bar t+1 starts a block
        for N in [30, 60]:
            acc = pd.DataFrame(0.0, index=R.index, columns=R.columns)
            cnt = pd.DataFrame(0.0, index=R.index, columns=R.columns)
            for k in range(1, N + 1):
                x = Bend.shift(6 * k - L)
                acc += x.fillna(0.0); cnt += x.notna()
            same = (acc / cnt).where(cnt >= N // 2)
            allm = R.rolling(6 * N, min_periods=3 * N).mean() * L
            for vn, sc in [("raw", same), ("excess", same - allm)]:
                Wt = cm.topn_ls(sc, U, 8).where(rebal, 0.0)
                W = drift_hold(Wt, R, rebal)
                name = f"A3_blk{blk_h}h_N{N}_{vn}_n8"
                o = cm.evaluate(W, "4h", name, "A3_session", N_TRIALS_A,
                                extra={"N": N, "variant": vn, "sel": "n8", "block_h": blk_h, "top_k": 50})
                rows.append(o)
                print(cm.short_line(o), flush=True)
            del acc, cnt, same
    pd.DataFrame(rows).to_csv(os.path.join(cm.HERE, "famA3_grid.csv"), index=False)
    print("done")
