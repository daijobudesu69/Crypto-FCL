"""Family B - PARTICIPATION / AVERAGE TRADE SIZE (round 2, inputs: qv, trades, close for residualising).

PRE-REGISTERED GRID (written before running; 34 configs counted)
------------------------------------------------------------------
Daily bars, per coin (trailing only):
  TC  = trades count, ATS = qv / trades (average trade size, USD)
  chg_x(h) = log(mean x over last h days) - log(mean x over the 60 days ending h days ago)
             (short window vs the non-overlapping prior 60-day baseline)
  B1 TCchg_h   trade-count surge                            h in {7,14,30}
  B2 ATSchg_h  average-trade-size change                    h in {7,14,30}
  B3 FOMO_h  = TCchg_h - ATSchg_h  (count up, size down; orthogonal to plain volume surge,
             because log volume change = TCchg + ATSchg; volume shock was ruled out in round 1)
  B4 ATSlvl  = log(mean ATS 30d), raw XS level; ATSrel = XS residual of log(mean ATS 30d) on
             log(mean qv 30d) (avg size given liquidity: "institutional share")
  each x direction {+,-}: B1-B3 = 18, B4 = 4                                           = 22
  B2/B3 momentum-residualised (XS OLS on ranks vs rank(ret_h), rank(ret_14)) x h x dir  = 12
Hypotheses: FOMO+ short / FOMO- long (retail crowding -> underperformance), ATSchg+ long (larger
  players accumulating). Direction "+" = long high signal.
Portfolio fixed: lab.hype_universe("1d", top_k=50), <10 names -> flat, quintile L/S (q=0.2),
  gross 1.0, daily rebalance, lab.HYPE_COST + funding.
TRANSLATION RISK: live, trade counts would come from HYPE candles ('n'), whose fill microstructure
  (one taker order split across makers, no Binance retail app flow, different lot/tick) differs from
  Binance's. trades are not in the hype_1d cache, so rank stability cannot be checked here - flagged.
"""
import os
import numpy as np, pandas as pd
import r2common as rc

LOG = os.path.join(rc.HERE, "famB_grid.csv")


def chg(x, h, base=60):
    s = x.rolling(h, min_periods=int(np.ceil(0.8 * h))).mean()
    b = x.rolling(base, min_periods=int(0.8 * base)).mean().shift(h)
    return np.log(s) - np.log(b)


def build_scores():
    p = rc.panels()
    T, Q = p["T"], p["Q"]
    ats = Q / T
    out = {}
    for h in (7, 14, 30):
        tc = chg(T, h); a = chg(ats, h)
        out[f"TCchg{h}"] = tc
        out[f"ATSchg{h}"] = a
        out[f"FOMO{h}"] = tc - a
    out["ATSlvl"] = np.log(ats.rolling(30, min_periods=24).mean())
    return out


def main():
    if os.path.exists(LOG):
        os.remove(LOG)
    Um = rc.U(50)
    p = rc.panels()
    S = build_scores()
    lq = np.log(p["Q"].rolling(30, min_periods=24).mean())
    S["ATSrel"] = rc.xs_resid(S["ATSlvl"], [lq], Um)
    n = 0
    names = [f"{k}{h}" for h in (7, 14, 30) for k in ("TCchg", "ATSchg", "FOMO")] + ["ATSlvl", "ATSrel"]
    for nm in names:
        for sgn, dname in ((1, "+"), (-1, "-")):
            W = rc.qls(sgn * S[nm], Um)
            row, _ = rc.evaluate(W, f"{nm}{dname}", "B_raw", LOG)
            n += 1
            print(rc.fmt(row), flush=True)
    for h in (7, 14, 30):
        for k in ("ATSchg", "FOMO"):
            r = rc.xs_resid(S[f"{k}{h}"], [rc.mom(h), rc.mom(14)], Um)
            for sgn, dname in ((1, "+"), (-1, "-")):
                W = rc.qls(sgn * r, Um)
                row, _ = rc.evaluate(W, f"{k}{h}res{dname}", "B_resid", LOG)
                n += 1
                print(rc.fmt(row), flush=True)
    print("configs run:", n)


if __name__ == "__main__":
    main()
