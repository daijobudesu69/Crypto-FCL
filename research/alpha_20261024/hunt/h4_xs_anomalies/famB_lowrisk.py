"""FAMILY B - LOW-RISK & LOTTERY ANOMALIES (1d bars, research period 2020-2024 via lab.py only)

PRE-REGISTERED GRID (written before any run; 36 configs)
--------------------------------------------------------
Common: universe = lab.universe("1d", min_hist_days=60, top_k=50) minus BTCUSDT (BTC is the factor:
        its beta is 1 and its idio-vol is 0 by construction). Signal must be non-NaN.
        Quintiles q=0.2. Score sign: LONG the "safe/boring" coins, SHORT the "risky/lottery" coins.
        Construction DN = $-neutral equal weight quintile L/S (lab.quantile_ls_weights, gross 1).
        Construction BN = beta-neutral (Frazzini-Pedersen): each leg equal weight, scaled so leg beta
                          = 0.5 vs BTC (shrunk 60d beta, 0.6*OLS + 0.4*XS median), leg gross capped 1.0.
        Rebalance R1 = daily targets, R7 = targets updated every 7 days (lab rebalance_every=7).
B1 BAB    : beta to BTC from daily returns, window {60, 90}         x {DN, BN} x {R1, R7}  = 8
B2 IVOL   : residual vol vs BTC (OLS in window), window {30, 60}   x {DN, BN} x {R1, R7}  = 8
B3 MAX    : MAX1 = max daily ret 30d, MAX5 = mean of top-5 daily rets 30d
                                                                   x {DN, BN} x {R1, R7}  = 8
B4 SKEW   : realized skew of daily rets {30d, 90d} and of 4h rets over 30d (180 bars, sampled at
            the 4h bar closing at 00:00 UTC = daily close)          x {DN, BN} x {R1, R7}  = 12
Primary selection metric: net Sharpe (0.07%/side + funding). Survivorship note: dead lottery coins
are missing from the 150-coin list -> short leg (high beta / IVOL / MAX / skew) is UNDERSTATED, the
long leg OVERSTATED.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import lab, rolling_beta, shrink_beta, beta_neutral_weights, evaluate, fmt, BTC

C = lab.load("close", "1d")
R = C.pct_change(fill_method=None)
U = lab.universe("1d", min_hist_days=60, top_k=50)
U[BTC] = False
rb = R[BTC]

beta60, rv60, _ = rolling_beta(R, rb, 60)
beta90, rv90, _ = rolling_beta(R, rb, 90)
_, rv30, _ = rolling_beta(R, rb, 30)
bshr = shrink_beta(beta60.where(U))           # beta used for neutralisation in BN

# 4h realized skewness over 30d, sampled at the daily close
R4 = lab.load("close", "4h").pct_change(fill_method=None)
sk4 = R4.rolling(180, min_periods=120).skew()
sk4 = sk4[sk4.index.hour == 20]
sk4.index = sk4.index.floor("1D")
sk4 = sk4.reindex(C.index)
del R4

def max5(x):
    a = np.sort(x[~np.isnan(x)])
    return a[-5:].mean() if len(a) >= 20 else np.nan

signals = {
    "BAB_b60": -beta60,
    "BAB_b90": -beta90,
    "IVOL_30": -np.sqrt(rv30),
    "IVOL_60": -np.sqrt(rv60),
    "MAX1_30": -R.rolling(30, min_periods=20).max(),
    "MAX5_30": -R.rolling(30, min_periods=20).apply(max5, raw=True),
    "SKEW_d30": -R.rolling(30, min_periods=20).skew(),
    "SKEW_d90": -R.rolling(90, min_periods=60).skew(),
    "SKEW_4h30": -sk4,
}

rows = []
for sname, s in signals.items():
    for cons in ["DN", "BN"]:
        if cons == "DN":
            W = lab.quantile_ls_weights(s, U & s.notna() & bshr.notna(), q=0.2)
        else:
            W = beta_neutral_weights(s, U, bshr, q=0.2)
        for reb in [1, 7]:
            Wr = W
            if reb > 1:
                keep = pd.Series(np.arange(len(W)) % reb == 0, index=W.index)
                Wr = W.where(keep, np.nan).ffill().fillna(0.0)
            name = f"B_{sname}_{cons}_R{reb}"
            row, res = evaluate(Wr, "1d", name, "B", U=U, R=R)
            rows.append(row)
            print(fmt(row), flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "famB_results.csv"), index=False)
print("\nconfigs run:", len(out))
print(out.sort_values("sharpe", ascending=False)[["name", "sharpe", "sharpe_cost2x", "sharpe_lag1", "sh_2020-21", "sh_2022", "sh_2023-24", "corr_btc", "corr_xsmom"]].head(12).round(2).to_string())
