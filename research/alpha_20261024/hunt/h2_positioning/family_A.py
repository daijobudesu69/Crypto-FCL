"""FAMILY A - SMART-vs-RETAIL POSITIONING (Binance metrics, 1d)

Hypothesis: retail ACCOUNT crowding (global_ls = share of all accounts net long / net short) is
contrarian - the crowd buys dips and is late; top traders' POSITION tilt (top_pos_ls, top-20% by
margin, weighted by position size) is informed. Who is on the other side: retail longs supply
liquidity to informed flow; persists because retail flow is sticky and the data is free but noisy.
Ratio levels differ structurally per coin -> trailing time-series transform first, then XS rank.

Data coverage (top-50, avg names/day): global_ls 2022-02 -> 2024 (~38-50), top_pos_ls/top_acct_ls
ONLY 2023-01 -> 2024 (~44-50; 2022 has gaps). So family A is a 2-3 year sample.

PRE-REGISTERED GRID (written before running):
 Universe top-50 PIT (lab.universe top_k=50), quintile L/S $-neutral gross 1, daily rebalance,
 flat when < 20 names have a valid score. Cost 0.07%/side + real funding.
 A1  -tsz(global_ls, L)                         L in {30, 90}        2
 A2  +tsz(top_pos_ls, L)                        L in {30, 90}        2
 A3  tsz(top_pos_ls,L) - tsz(global_ls,L)       L in {30, 90}        2
 A4  -dlog(global_ls, k)                        k in {1, 3, 7}       3
 A5  dlog(top_pos_ls,k) - dlog(global_ls,k)     k in {1, 3, 7}       3
 A6  +tsz(log(top_pos_ls/top_acct_ls), L)       L in {30, 90}        2   (whale size vs account tilt)
 A7  -log(global_ls) raw XS level (control)                          1
                                                         subtotal   15
 Robustness on best 2 by IS Sharpe: rebalance_every=3, top_k=100, q=0.33, momentum-neutral
 (signal residualised on XS rank of 14d return)                       8
 TOTAL <= 23 (+ any sign flip is counted as an extra trial)
"""
from h2lib import *

U = lab.universe("1d", top_k=50)
C = lab.load("close", "1d")
G = lab.load("global_ls", "1d"); T = lab.load("top_pos_ls", "1d"); A = lab.load("top_acct_ls", "1d")
T = T.where(T > 0); A = A.where(A > 0); G = G.where(G > 0)
F = lab.load("fund", "1d")
R1 = C.pct_change(fill_method=None).shift(-1)           # next-day return (target only)
R7 = (C.shift(-7) / C - 1)                              # next-7d return (target only)
ret7 = C / C.shift(7) - 1; ret14 = C / C.shift(14) - 1
fund7 = F.rolling(7, min_periods=5).sum()

sig = {}
for L in (30, 90):
    sig[f"A1_-zG{L}"] = -tsz(G, L)
    sig[f"A2_zT{L}"] = tsz(T, L)
    sig[f"A3_zT-zG{L}"] = tsz(T, L) - tsz(G, L)
    sig[f"A6_zTpos/acct{L}"] = tsz(np.log(T / A), L)
for k in (1, 3, 7):
    sig[f"A4_-dG{k}"] = -lchg(G, k)
    sig[f"A5_dT-dG{k}"] = lchg(T, k) - lchg(G, k)
sig["A7_-levelG"] = -np.log(G)

print("Diagnostics: mean daily XS Spearman IC vs next-1d / next-7d return, corr with past 7d return & 7d funding")
diag = {}
for n, s in sig.items():
    ic1 = xs_ic(s, R1, U); ic7 = xs_ic(s, R7, U)
    diag[n] = {"IC1": ic1[0], "t1": ic1[1], "IC7": ic7[0], "t7": ic7[1], "days": ic1[2],
               "c_ret7": xs_corr(s, ret7, U), "c_fund7": xs_corr(s, fund7, U)}
print(pd.DataFrame(diag).T.round(3).to_string())

rows = {}
for n, s in sig.items():
    W = qls(s, U)
    row, res = evaluate(n, "A", W)
    rows[n] = row
    print(fmt(row))
pd.DataFrame(rows).T.to_csv(os.path.join(HERE, "family_A_grid.csv"))
