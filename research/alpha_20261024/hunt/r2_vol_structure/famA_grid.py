"""FAMILY A - ASYMMETRIC RISK PREMIA (downside beta / upside beta / coskewness vs BTC)

PRE-REGISTERED GRID (written before any run, 32 configs):
  signal    : bminus  (beta on days BTC<0, Ang-Chen-Xing downside beta)
              bplus   (beta on days BTC>0)
              bdiff   (bminus - bplus, asymmetry)
              coskew  (Harvey-Siddique standardized coskewness with BTC, CAPM residual)
  window    : 90, 180 trailing daily returns (min_periods 67%; conditional betas need >=30% of W obs)
  direction : TB   = textbook: long high bminus | long LOW bplus | long high bdiff | long NEGATIVE coskew
              FLIP = opposite sign
  construct : DN = $-neutral quintile L/S (EW legs, 0.5 gross each)
              BN = beta-neutral: same quintile legs, leg gross ~ 1/avg(plain beta_W), total gross 1
  fixed     : universe lab.hype_universe('1d', top_k=50) minus BTC (BTC beta==1 by construction),
              rebalance every 7 days (weights held, drift re-traded daily by lab), cost lab.HYPE_COST,
              funding charged as cost. Stress: 2x cost, extra_lag=1.
  4 x 2 x 2 x 2 = 32 configs. Robustness (top_k 30/75, 16-pos, daily rebalance) only for the best few,
  counted separately in famA_robust.py.
Inputs: close (1d) only. Signal at row t uses returns up to close t (trailing rolling windows).
"""
import os
import pandas as pd, numpy as np
from common import HERE, universe, ls_weights, bn_weights, evaluate, fmt_row
from signals import beta_family

TB_SIGN = {"bminus": +1, "bplus": -1, "bdiff": +1, "coskew": -1}
N_TRIALS = 32

if __name__ == "__main__":
    U = universe(50, ex_btc=True)
    rows = []
    for W in (90, 180):
        S = beta_family(W)
        for sig in ("bminus", "bplus", "bdiff", "coskew"):
            for dname, dsign in (("TB", 1), ("FLIP", -1)):
                score = S[sig] * TB_SIGN[sig] * dsign
                for cons in ("DN", "BN"):
                    if cons == "DN":
                        Wt = ls_weights(score, U, q=0.2)
                    else:
                        Wt = bn_weights(score, U, S["beta"], q=0.2)
                    name = f"A_{sig}_{W}_{dname}_{cons}"
                    o, res = evaluate(Wt, name, rebalance=7, mask=U, n_trials=N_TRIALS)
                    o.update({"signal": sig, "window": W, "dir": dname, "cons": cons})
                    rows.append(o)
                    print(name.ljust(26), fmt_row(o), f"alpha_t={o['alpha_t']:.2f} cXS={o['corr_xsmom']:.2f}",
                          flush=True)
        del S
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "famA_grid.csv"), index=False)
    print(df.sort_values("sharpe", ascending=False)[["name", "sharpe", "sh_2020-21", "sh_2022", "sh_2023-24",
          "sharpe_cost2x", "sharpe_lag1", "alpha_t", "corr_xsmom", "avg_net"]].round(2).to_string())
