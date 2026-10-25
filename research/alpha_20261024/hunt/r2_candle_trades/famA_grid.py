"""Family A - CANDLE SHAPE / ACCUMULATION (round 2, HYPE-candle inputs only: open/high/low/close).

PRE-REGISTERED GRID (written before running; 42 configs counted)
------------------------------------------------------------------
Daily bars, per coin (trailing only, row t known at close t):
  rng = H-L (bars with H<=L or no trades -> NaN)
  CLV = ((C-L)-(H-C))/rng          close-location value, in [-1,1]
  UW  = (H-max(O,C))/rng           upper-wick share
  LW  = (min(O,C)-L)/rng           lower-wick share
  WI  = LW-UW                      wick imbalance (= CLV minus signed body/range -> strips the
                                   open-to-close return, i.e. the momentum-like part of CLV)
  feature_N = rolling mean over N days (min_periods = 0.8N)
  RPOS_N = (C - min L_N)/(max H_N - min L_N)   range position of close (stochastic %K; momentum proxy)
Grid:
  1) raw: feature in {CLV, UW, LW, WI, RPOS} x N in {5,10,20} x direction in {+,-}       = 30
  2) momentum-residualised: CLV_N and WI_N residualised each day (XS OLS on ranks) against
     rank(ret_N) and rank(ret_14)  x N in {5,10,20} x direction {+,-}                     = 12
  "+" = long high feature / short low feature.
Portfolio (fixed, not tuned): lab.hype_universe("1d", top_k=50), days with <10 names flat,
  quintile long/short equal weight (lab.quantile_ls_weights q=0.2), gross 1.0, daily rebalance,
  cost lab.HYPE_COST per side + funding as cost.
Robustness only for the best per family (not selection): 2x cost, extra_lag=1, legs, top_k 30/75,
  8L/8S (16-position) version, neighbour plateau.
Hypothesis: persistent closes near the high (CLV>0, small upper wicks) = patient buyers absorbing
  supply (continuation), or = late buying exhaustion (reversal). Lower wicks = dip demand.
"""
import os
import numpy as np, pandas as pd
import r2common as rc

LOG = os.path.join(rc.HERE, "famA_grid.csv")


def features():
    p = rc.panels()
    O, H, L, C = p["O"], p["H"], p["L"], p["C"]
    rng = H - L
    clv = ((C - L) - (H - C)) / rng
    uw = (H - np.maximum(O, C)) / rng
    lw = (np.minimum(O, C) - L) / rng
    wi = lw - uw
    return {"CLV": clv, "UW": uw, "LW": lw, "WI": wi}


def build_scores():
    p = rc.panels()
    f = features()
    out = {}
    for N in (5, 10, 20):
        mp = int(np.ceil(0.8 * N))
        for k, v in f.items():
            out[f"{k}{N}"] = v.rolling(N, min_periods=mp).mean()
        lo = p["L"].rolling(N, min_periods=mp).min()
        hi = p["H"].rolling(N, min_periods=mp).max()
        out[f"RPOS{N}"] = (p["C"] - lo) / (hi - lo)
    return out


def main():
    if os.path.exists(LOG):
        os.remove(LOG)
    Um = rc.U(50)
    S = build_scores()
    n = 0
    for N in (5, 10, 20):
        for k in ("CLV", "UW", "LW", "WI", "RPOS"):
            s = S[f"{k}{N}"]
            for sgn, dname in ((1, "+"), (-1, "-")):
                W = rc.qls(sgn * s, Um)
                row, _ = rc.evaluate(W, f"{k}{N}{dname}", "A_raw", LOG)
                n += 1
                print(rc.fmt(row), flush=True)
    for N in (5, 10, 20):
        for k in ("CLV", "WI"):
            r = rc.xs_resid(S[f"{k}{N}"], [rc.mom(N), rc.mom(14)], Um)
            for sgn, dname in ((1, "+"), (-1, "-")):
                W = rc.qls(sgn * r, Um)
                row, _ = rc.evaluate(W, f"{k}{N}res{dname}", "A_resid", LOG)
                n += 1
                print(rc.fmt(row), flush=True)
    print("configs run:", n)


if __name__ == "__main__":
    main()
