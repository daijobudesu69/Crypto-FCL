"""POST-HOC (declared as such, counted): A9 level of top_pos_ls was strongly NEGATIVE (-1.54). Unified
re-interpretation: 'crowded long positioning, by account count (global_ls) or by top-trader size
(top_pos_ls), is contrarian' - i.e. A9-flip is the top-trader version of A7. 3 counted trials:
  A9f   -mean14(log top_pos_ls)
  A10   -mean14(log top_pos_ls + log global_ls)/2  (consensus crowding)
  A9f|fundneutral  (resid on XS rank of 7d funding) - is it just funding?
Diagnostics: XS correlations of the levels with each other, size, 60d momentum, funding."""
from h2lib import *
U = lab.universe("1d", top_k=50)
C = lab.load("close", "1d"); QV = lab.load("qv", "1d"); F = lab.load("fund", "1d")
G = lab.load("global_ls", "1d"); T = lab.load("top_pos_ls", "1d")
G = np.log(G.where(G > 0)); T = np.log(T.where(T > 0))
sm = lambda X: X.rolling(14, min_periods=7).mean()
sT, sG = sm(T), sm(G)
fund7 = F.rolling(7, min_periods=5).sum(); fund30 = F.rolling(30, min_periods=15).sum()
size = np.log(QV.rolling(30, min_periods=15).median()); ret60 = C / C.shift(60) - 1; ret14 = C / C.shift(14) - 1
for n, x in [("logG14", sG), ("fund7", fund7), ("fund30", fund30), ("size", size), ("ret14", ret14), ("ret60", ret60)]:
    print(f"xs corr(logT14, {n}) = {xs_corr(sT, x, U):.3f}   xs corr(logG14, {n}) = {xs_corr(sG, x, U):.3f}")
for n, s in [("A9f_-lvl_Tpos", -sT), ("A10_-lvl_(T+G)/2", -(sT + sG) / 2), ("A9f|fundneutral", xs_resid(-sT, fund7, U))]:
    row, res = evaluate(n, "A", qls(s, U), n_note="post-hoc"); print(fmt(row))
    print("    yearly:", lab.yearly(res["df"]["r"]).round(3)["ret"].to_dict(), " t_NW", round(res["metrics"]["t_nw"], 2))
