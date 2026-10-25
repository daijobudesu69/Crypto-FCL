"""POST-HOC single config (counted as 1 extra trial in family B; total B effective trials = 86).

Motivation (from the leg decomposition of BR_SKEWrev_4h30_DN_R1, not from a new search): the
short leg - coins whose 4h returns over the last 30d are most NEGATIVELY skewed - underperformed the
EW universe in every year 2020-2024 (excess Sharpe 3.3 / 2.7 / 0.4 / 1.0 / 0.5), while the long
(positive-skew, "lottery") leg was unstable. Tradable expression with few positions:
  SHORT bottom quintile of 4h-skew-30d (equal weight, leg gross 0.5), universe top50 ex BTC,
  LONG BTC sized to the short leg's shrunk 60d beta (beta-neutral ex ante), daily rebalance.
Survivorship: dead crash coins are missing -> this short leg is, if anything, UNDERSTATED.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import lab, rolling_beta, shrink_beta, evaluate, fmt, BTC, controls

HERE = os.path.dirname(os.path.abspath(__file__))
C = lab.load("close", "1d"); R = C.pct_change(fill_method=None)
U = lab.universe("1d", min_hist_days=60, top_k=50); U[BTC] = False
b60, _, _ = rolling_beta(R, R[BTC], 60); bshr = shrink_beta(b60.where(U))
x = pd.read_pickle(os.path.join(HERE, "famB_rev_signals.pkl"))["SKEWrev_4h30"]
Wls = lab.quantile_ls_weights(x, U & x.notna() & bshr.notna(), q=0.2)
W = Wls.clip(upper=0)                                   # short leg only, gross 0.5
W[BTC] = (-W * bshr.fillna(1.0)).sum(axis=1)            # BTC hedge = short-leg beta
row, res = evaluate(W, "1d", "BS_short_negskew4h_BTChedge", "B-post", U=U, R=R)
print(fmt(row))
print(lab.yearly(res["df"]["r"]).round(3).T.to_string())
j = pd.concat([res["df"]["r"].rename("y"), controls()], axis=1).dropna()
X = np.column_stack([np.ones(len(j)), j[["btc", "xsmom14"]].values])
b, *_ = np.linalg.lstsq(X, j["y"].values, rcond=None)
e = j["y"].values - X @ b
print(f"spanning: alpha/yr {b[0]*365:+.1%} t~{lab.newey_west_t(e + b[0], 10):.2f} b_btc {b[1]:+.2f} b_xsmom {b[2]:+.2f}")
m = res["metrics"]
print("DSR(n=86):", round(lab.deflated_sharpe(m["sharpe"], 86, m["days"], m["skew"], m["kurt"] + 3), 3),
      "DSR(n=170):", round(lab.deflated_sharpe(m["sharpe"], 170, m["days"], m["skew"], m["kurt"] + 3), 3))
print({k: row[k] for k in ["min_equity_usd", "min_equity_usd_p90", "avg_gross", "avg_net", "turnover_per_year"]})
res["df"]["r"].to_frame("r").to_parquet(os.path.join(HERE, "BS_short_negskew4h_daily.parquet"))
