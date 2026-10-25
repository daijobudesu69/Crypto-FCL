"""FAMILY B - CROSS-SECTIONAL VOLATILITY STRUCTURE (market-neutral, not time-series breakouts)

PRE-REGISTERED GRID (written before any run, 32 configs):
  signal (8):
    vts5      sqrt(mean RV_d, 5d) / sqrt(mean RV_d, 60d), RV_d = sum of squared 4h log returns in the UTC day
    vts10     same with 10d short window
    atr5_50   ATR%(5) / ATR%(50)      (true range / prev close)
    atr20_100 ATR%(20) / ATR%(100)    (slow compression regime)
    vov30     std over 30d of log(Parkinson daily range log(H/L))   (vol-of-vol, scale free)
    vov90     same over 90d
    pkcc30    Parkinson vol / close-to-close vol over 30d (intraday noise ratio)
    pkcc90    same over 90d
  direction (2):
    TB  = textbook: long COMPRESSED / short EXPANDED for vts*, atr* (Vasquez-style term-structure:
          vol spikes = attention shocks that later underperform); long LOW vol-of-vol (Baltussen et al.
          2018 'unknown unknowns'); long HIGH pkcc (Corwin-Schultz high-low illiquidity premium)
    FLIP = opposite sign
  rebalance (2): every 1 day, every 7 days
  fixed: universe lab.hype_universe('1d', top_k=50) (BTC included), $-neutral quintile L/S (EW legs, 0.5 gross
         each), cost lab.HYPE_COST, funding charged as cost. Stress: 2x cost, extra_lag=1.
  8 x 2 x 2 = 32 configs. Robustness for the best few (top_k 30/75, 16-pos, beta-neutral, rv60 neutral)
  counted separately in famB_robust.py.
Inputs: 1d open/high/low/close + 4h close only. All windows trailing.
"""
import os
import pandas as pd
from common import HERE, universe, ls_weights, evaluate, fmt_row
from signals import vol_structure

TB_SIGN = {"vts5": -1, "vts10": -1, "atr5_50": -1, "atr20_100": -1, "vov30": -1, "vov90": -1,
           "pkcc30": +1, "pkcc90": +1}
N_TRIALS = 32

if __name__ == "__main__":
    U = universe(50)
    S = vol_structure()
    rows = []
    for sig in TB_SIGN:
        for dname, dsign in (("TB", 1), ("FLIP", -1)):
            score = S[sig] * TB_SIGN[sig] * dsign
            Wt = ls_weights(score, U, q=0.2)
            for reb in (1, 7):
                name = f"B_{sig}_{dname}_R{reb}"
                o, res = evaluate(Wt, name, rebalance=reb, mask=U, n_trials=N_TRIALS)
                o.update({"signal": sig, "dir": dname, "reb": reb})
                rows.append(o)
                print(name.ljust(24), fmt_row(o), f"alpha_t={o['alpha_t']:.2f} cXS={o['corr_xsmom']:.2f}"
                      f" cBTC={o['corr_btc']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "famB_grid.csv"), index=False)
    print(df.sort_values("sharpe", ascending=False)[["name", "sharpe", "sh_2020-21", "sh_2022", "sh_2023-24",
          "sharpe_cost2x", "sharpe_lag1", "alpha_t", "corr_xsmom", "corr_btc", "long_leg_xs_sharpe",
          "short_leg_xs_sharpe"]].round(2).to_string())
