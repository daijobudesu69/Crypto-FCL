"""FAMILY A follow-up - the pre-registered reversal grid (40 configs) found CONTINUATION, not reversal,
at 1-5 day horizons, strongest for high-abnormal-volume moves (A2 HIGHAV). This checks whether the
flipped sign (short-horizon, volume-confirmed momentum) is anything more than the existing 14d
XS-momentum factor. Sign choice is EX POST -> counts as extra trials (A family = 40*2 + 4).

PRE-REGISTERED (4 configs, 1d, universe top50, same construction as famA):
  AF1 = -A1_L3_RESID_D    3d residual momentum, quintile, daily
  AF2 = -A1_L3_RESID_S3   same, 3-day overlapping books
  AF3 = -A2_L3_RAW_HIGHAV  3d raw momentum inside the high-abnormal-volume half (q=0.3)
  AF4 = -A2_L3_RESID_HIGHAV 3d residual momentum inside the high-abnormal-volume half (q=0.3)
Decisive test: spanning regression on BTC + 14d XS-momentum control (alpha t-stat, NW lag 10).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import lab, rolling_beta, evaluate, fmt, BTC, controls

HERE = os.path.dirname(os.path.abspath(__file__))
C = lab.load("close", "1d"); R = C.pct_change(fill_method=None); qv = lab.load("qv", "1d")
U = lab.universe("1d", min_hist_days=60, top_k=50)
beta, _, _ = rolling_beta(R, R[BTC], 60)
eps = R.sub(beta.mul(R[BTC], axis=0))
L = 3
raw = C / C.shift(L) - 1
res_ = eps.rolling(L, min_periods=L).sum()
AV = np.log(qv.rolling(L).sum() / (L * qv.shift(L).rolling(30, min_periods=20).median())).replace([np.inf, -np.inf], np.nan)


def hi_half(score):
    m = U & score.notna() & AV.notna()
    return lab.quantile_ls_weights(score, m & (lab.xs_rank(AV, m) > 0.5), q=0.3)


Wq = lab.quantile_ls_weights(res_, U & res_.notna(), q=0.2)
books = {"AF1_mom3_RESID_D": Wq,
         "AF2_mom3_RESID_S3": Wq.fillna(0).rolling(3, min_periods=1).mean(),
         "AF3_mom3_RAW_HIGHAV": hi_half(raw),
         "AF4_mom3_RESID_HIGHAV": hi_half(res_)}
ctrl = controls()
rows = []
for nm, W in books.items():
    row, r = evaluate(W, "1d", nm, "A-flip", U=U, R=R)
    rows.append(row); print(fmt(row), flush=True)
    j = pd.concat([r["df"]["r"].rename("y"), ctrl], axis=1).dropna()
    X = np.column_stack([np.ones(len(j)), j[["btc", "xsmom14"]].values])
    b, *_ = np.linalg.lstsq(X, j["y"].values, rcond=None)
    e = j["y"].values - X @ b
    print(f"    spanning: alpha/yr {b[0]*365:+.1%} t~{lab.newey_west_t(e + b[0], 10):.2f}  b_btc {b[1]:+.2f}  b_xsmom14 {b[2]:+.2f}")
pd.DataFrame(rows).to_csv(os.path.join(HERE, "famA_flip_results.csv"), index=False)
