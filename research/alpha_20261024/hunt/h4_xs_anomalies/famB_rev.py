"""FAMILY B follow-up - "LOTTERY PREMIUM" (reversed sign), 1d, research period 2020-2024 via lab.py only

WHY THIS EXISTS: the pre-registered famB grid (36 configs) REJECTED the low-risk/lottery discount:
every IVOL/MAX/SKEW config had negative net Sharpe in the textbook direction (long boring / short
lottery), consistently across windows, constructions and rebalance speeds. The sign choice below is
therefore EX POST -> for deflated-Sharpe purposes the 36 famB configs count twice (72) plus these.

PRE-REGISTERED FOLLOW-UP GRID (written before running; 10 configs + 3 robustness)
Common: universe top_k=50 ex BTC, q=0.2, DN = $-neutral equal-weight quintiles, BN = beta-neutral legs
        (same helpers as famB). Long = HIGH score.
F1 SKEWrev_d30  : +skew of daily returns, 30d                       x DN x {R1, R7}       = 2
F2 SKEWrev_4h30 : +skew of 4h returns over 30d (180 bars)            x DN x {R1, R7}       = 2
F3 MAXrev5_30   : +mean of the 5 highest daily returns in 30d        x DN x {R1, R7}       = 2
F4 MIN5_30      : +mean of the 5 lowest daily returns in 30d (long mild-worst-day coins,
                  short coins with the most severe crash days) -> left- vs right-tail split
                                                                     x DN x {R1, R7}       = 2
F5 SKEWrev_d30  : BN x {R1, R7}                                                            = 2
Robustness on whichever of F1-F5 has the best net Sharpe (NOT used for selection):
  universe top_k=30, top_k=100, q=0.3                                                      = 3
Diagnostics: spanning regression of daily net returns on BTC, 14d XS-momentum control and a
3-day residual-momentum control (flipped A1_L3_RESID_D); leg decomposition vs EW universe.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import lab, rolling_beta, shrink_beta, beta_neutral_weights, evaluate, fmt, BTC, controls

HERE = os.path.dirname(os.path.abspath(__file__))
C = lab.load("close", "1d")
R = C.pct_change(fill_method=None)
U = lab.universe("1d", min_hist_days=60, top_k=50); U[BTC] = False
beta60, _, _ = rolling_beta(R, R[BTC], 60)
bshr = shrink_beta(beta60.where(U))

R4 = lab.load("close", "4h").pct_change(fill_method=None)
sk4 = R4.rolling(180, min_periods=120).skew()
sk4 = sk4[sk4.index.hour == 20]; sk4.index = sk4.index.floor("1D"); sk4 = sk4.reindex(C.index)
del R4


def topk_mean(x, k, hi=True):
    a = np.sort(x[~np.isnan(x)])
    if len(a) < 20:
        return np.nan
    return a[-k:].mean() if hi else a[:k].mean()


sig = {
    "SKEWrev_d30": R.rolling(30, min_periods=20).skew(),
    "SKEWrev_4h30": sk4,
    "MAXrev5_30": R.rolling(30, min_periods=20).apply(lambda x: topk_mean(x, 5, True), raw=True),
    "MIN5_30": R.rolling(30, min_periods=20).apply(lambda x: topk_mean(x, 5, False), raw=True),
}
sig_path = os.path.join(HERE, "famB_rev_signals.pkl")
pd.to_pickle(sig, sig_path)


def hold(W, reb):
    if reb == 1:
        return W
    keep = pd.Series(np.arange(len(W)) % reb == 0, index=W.index)
    return W.where(keep, np.nan).ffill().fillna(0.0)


rows, rets, params = [], {}, {}
plan = [(s, "DN", r) for s in ["SKEWrev_d30", "SKEWrev_4h30", "MAXrev5_30", "MIN5_30"] for r in [1, 7]]
plan += [("SKEWrev_d30", "BN", 1), ("SKEWrev_d30", "BN", 7)]
for s, cons, reb in plan:
    x = sig[s]
    if cons == "DN":
        W = lab.quantile_ls_weights(x, U & x.notna() & bshr.notna(), q=0.2)
    else:
        W = beta_neutral_weights(x, U, bshr, q=0.2)
    W = hold(W, reb)
    name = f"BR_{s}_{cons}_R{reb}"
    params[name] = (s, cons, reb)
    row, res = evaluate(W, "1d", name, "B-rev", U=U, R=R)
    rows.append(row); rets[name] = res["df"]["r"]
    print(fmt(row), flush=True)

out = pd.DataFrame(rows)
best = out.sort_values("sharpe", ascending=False).iloc[0]["name"]
print("\nbest follow-up config:", best)
bs, bc, br = params[best]
x = sig[bs]
for tag, k, q in [("U30", 30, 0.2), ("U100", 100, 0.2), ("q30", 50, 0.3)]:
    Uk = lab.universe("1d", min_hist_days=60, top_k=k); Uk[BTC] = False
    if bc == "DN":
        W = lab.quantile_ls_weights(x, Uk & x.notna() & bshr.notna(), q=q)
    else:
        W = beta_neutral_weights(x, Uk, shrink_beta(beta60.where(Uk)), q=q)
    W = hold(W, br)
    row, res = evaluate(W, "1d", f"{best}_rob_{tag}", "B-rev-rob", U=Uk, R=R)
    rows.append(row); rets[row["name"]] = res["df"]["r"]
    print(fmt(row), flush=True)

# --- spanning regression vs controls (BTC, 14d XS mom, 3d residual momentum)
eps = R.sub(beta60.mul(R[BTC], axis=0))
U50 = lab.universe("1d", min_hist_days=60, top_k=50)
m3 = eps.rolling(3, min_periods=3).sum()
stm = lab.backtest(lab.quantile_ls_weights(m3, U50 & m3.notna(), q=0.2), "1d")["df"]["r"].rename("stmom3")
ctrl = controls().join(stm, how="outer")
print("\nSpanning regressions (daily, OLS, NW t-stats lag 10):")
for name, r in rets.items():
    j = pd.concat([r.rename("y"), ctrl], axis=1).dropna()
    X = np.column_stack([np.ones(len(j)), j[["btc", "xsmom14", "stmom3"]].values])
    b, *_ = np.linalg.lstsq(X, j["y"].values, rcond=None)
    e = j["y"].values - X @ b
    a_t = lab.newey_west_t(e + b[0], lags=10)
    print(f"{name:42s} alpha/yr {b[0]*365:+.1%} (t~{a_t:.2f})  b_btc {b[1]:+.2f} b_xsmom {b[2]:+.2f} b_stmom3 {b[3]:+.2f}  "
          f"corr_stmom3 {j['y'].corr(j['stmom3']):+.2f}")
out.to_csv(os.path.join(HERE, "famB_rev_results.csv"), index=False)
pd.DataFrame(rets).to_parquet(os.path.join(HERE, "famB_rev_daily_returns.parquet"))
print("configs run:", len(out))
