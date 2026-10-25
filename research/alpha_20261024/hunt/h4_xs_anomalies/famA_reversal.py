"""FAMILY A - SHORT-TERM REVERSAL (liquidity provision), research period 2020-2024 via lab.py only

PRE-REGISTERED GRID (written before any run; 40 configs)
--------------------------------------------------------
Score = minus formation return (long losers, short winners). Quintile L/S q=0.2, $-neutral, gross 1
(lab.quantile_ls_weights). Universe = lab.universe(tf, min_hist_days=60, top_k=50), point in time.
ret type RAW   = simple compounded return over the formation window.
ret type RESID = sum over the window of (r_i - beta_i * r_BTC), beta = trailing 60-day OLS beta
                 to BTC (1d: 60 bars, 4h: 360 bars), known at the close of the signal bar.
exec D/B = trade the fresh target every bar; S3 / S6 = overlapping portfolios: target = mean of the
           last 3 daily (resp. 6 four-hour) target books -> lower turnover, older signal.
A1 1d : L in {1,2,3,5} days x {RAW, RESID} x {D, S3}                                    = 16
A2 1d volume-conditioned double sort: L in {1,3} x {RAW, RESID} x {LOWAV, HIGHAV}         =  8
        AV = log(qv summed over the L formation days / (L * median daily qv of the 30 days before
        the formation window)). Each day split the universe at the XS median of AV; inside the
        chosen half go long bottom 30% / short top 30% of formation return (q=0.3), gross 1, exec D.
        Hypothesis (CGW 1993 / Llorente et al. 2002 / task brief): low-AV moves revert (liquidity),
        high-AV moves continue (information) -> LOWAV reversal Sharpe > HIGHAV reversal Sharpe.
A3 4h : formation h in {1,3,6} bars (4h/12h/24h) x {RAW, RESID} x {B, S6}               = 12
A4 4h volume-conditioned: h in {3,6} x RAW x {LOWAV, HIGHAV}, AV vs median of prior 180 bars,
        exec B                                                                            =  4
Primary metric: net Sharpe (0.07%/side + funding). Cost drag reported explicitly.
Survivorship: long-loser leg buys coins after drops; dead coins (drops that never recovered) are
missing -> long leg overstated.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import lab, rolling_beta, evaluate, fmt, BTC

HERE = os.path.dirname(os.path.abspath(__file__))
part = sys.argv[1] if len(sys.argv) > 1 else "1d"


def smooth(W, k):
    return W.fillna(0.0).rolling(k, min_periods=1).mean()


def dbl_sort_weights(score, AV, U, half, q=0.3):
    m = U & score.notna() & AV.notna()
    avr = lab.xs_rank(AV, m)
    mh = m & ((avr <= 0.5) if half == "LOWAV" else (avr > 0.5))
    return lab.quantile_ls_weights(score, mh, q=q)


rows = []
if part == "1d":
    tf = "1d"
    C = lab.load("close", tf); R = C.pct_change(fill_method=None)
    qv = lab.load("qv", tf)
    U = lab.universe(tf, min_hist_days=60, top_k=50)
    beta, _, _ = rolling_beta(R, R[BTC], 60)
    eps = R.sub(beta.mul(R[BTC], axis=0))
    for L in [1, 2, 3, 5]:
        raw = C / C.shift(L) - 1
        res_ = eps.rolling(L, min_periods=L).sum()
        for rt, f in [("RAW", raw), ("RESID", res_)]:
            s = -f
            W = lab.quantile_ls_weights(s, U & s.notna(), q=0.2)
            for ex in ["D", "S3"]:
                Wx = W if ex == "D" else smooth(W, 3)
                row, _ = evaluate(Wx, tf, f"A1_L{L}_{rt}_{ex}", "A", U=U, R=R)
                rows.append(row); print(fmt(row), flush=True)
    for L in [1, 3]:
        AV = np.log(qv.rolling(L).sum() / (L * qv.shift(L).rolling(30, min_periods=20).median()))
        AV = AV.replace([np.inf, -np.inf], np.nan)
        raw = C / C.shift(L) - 1
        res_ = eps.rolling(L, min_periods=L).sum()
        for rt, f in [("RAW", raw), ("RESID", res_)]:
            for half in ["LOWAV", "HIGHAV"]:
                W = dbl_sort_weights(-f, AV, U, half)
                row, _ = evaluate(W, tf, f"A2_L{L}_{rt}_{half}", "A", U=U, R=R)
                rows.append(row); print(fmt(row), flush=True)
else:
    tf = "4h"
    C = lab.load("close", tf); R = C.pct_change(fill_method=None)
    qv = lab.load("qv", tf)
    U = lab.universe(tf, min_hist_days=60, top_k=50)
    beta, _, _ = rolling_beta(R, R[BTC], 360)
    eps = R.sub(beta.mul(R[BTC], axis=0))
    del beta
    for h in [1, 3, 6]:
        raw = C / C.shift(h) - 1
        res_ = eps.rolling(h, min_periods=h).sum()
        for rt, f in [("RAW", raw), ("RESID", res_)]:
            s = -f
            W = lab.quantile_ls_weights(s, U & s.notna(), q=0.2)
            for ex in ["B", "S6"]:
                Wx = W if ex == "B" else smooth(W, 6)
                row, _ = evaluate(Wx, tf, f"A3_h{h}_{rt}_{ex}", "A", U=U, R=R)
                rows.append(row); print(fmt(row), flush=True)
    for h in [3, 6]:
        AV = np.log(qv.rolling(h).sum() / (h * qv.shift(h).rolling(180, min_periods=120).median()))
        AV = AV.replace([np.inf, -np.inf], np.nan)
        raw = C / C.shift(h) - 1
        for half in ["LOWAV", "HIGHAV"]:
            W = dbl_sort_weights(-raw, AV, U, half)
            row, _ = evaluate(W, tf, f"A4_h{h}_RAW_{half}", "A", U=U, R=R)
            rows.append(row); print(fmt(row), flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, f"famA_results_{part}.csv"), index=False)
print("\nconfigs run:", len(out))
