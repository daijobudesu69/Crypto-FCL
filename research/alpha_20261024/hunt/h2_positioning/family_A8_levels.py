"""A8/A9 (pre-registered BEFORE running): level-form test of 'top traders informed vs retail', in the
same form that worked for A7 (14d trailing mean of log ratio, daily rebalance, top-50 quintile L/S).
Data only 2023-01 -> 2024 for top-trader fields.
  A8p  mean14(log top_pos_ls - log global_ls)   (top-trader POSITION tilt minus retail account tilt)
  A8a  mean14(log top_acct_ls - log global_ls)  (top-trader ACCOUNT tilt minus retail)
  A9   mean14(log top_pos_ls)                   (top-trader position tilt alone)
3 configs. Reference: A7 (-mean14 log global_ls) restricted to the same 2023-24 window (not a new trial)."""
from h2lib import *
U = lab.universe("1d", top_k=50)
G = lab.load("global_ls", "1d"); T = lab.load("top_pos_ls", "1d"); A = lab.load("top_acct_ls", "1d")
G = np.log(G.where(G > 0)); T = np.log(T.where(T > 0)); A = np.log(A.where(A > 0))
sm = lambda X: X.rolling(14, min_periods=7).mean()
for n, s in [("A8p_lvl_Tpos-G", sm(T - G)), ("A8a_lvl_Tacct-G", sm(A - G)), ("A9_lvl_Tpos", sm(T))]:
    row, _ = evaluate(n, "A", qls(s, U)); print(fmt(row))
row, _ = evaluate("REF_A7_S14_2023-24", "A-ref", qls(-sm(G), U), start="2023-01-01"); print(fmt(row))
