"""A7 (-log global_ls level) confound check: is it a size/liquidity or funding factor?
Diagnostics + 2 counted trials: A7 residualised on size (log 30d median qv) and on size+mom14 (2-step)."""
from h2lib import *
U = lab.universe("1d", top_k=50)
C = lab.load("close", "1d"); QV = lab.load("qv", "1d"); F = lab.load("fund", "1d")
G = lab.load("global_ls", "1d"); G = G.where(G > 0)
size = np.log(QV.rolling(30, min_periods=15).median())
ret14 = C / C.shift(14) - 1; ret60 = C / C.shift(60) - 1
fund7 = F.rolling(7, min_periods=5).sum(); fund30 = F.rolling(30, min_periods=15).sum()
s = -np.log(G)
for n, x in [("size", size), ("ret14", ret14), ("ret60", ret60), ("fund7", fund7), ("fund30", fund30)]:
    print(f"xs corr(-logG, {n}) = {xs_corr(s, x, U):.3f}")
# pure size control factor (long big, short small)
row, _ = evaluate("CTRL_size_long_big", "A-ctrl", qls(size, U)); print(fmt(row))
row, _ = evaluate("CTRL_mom60", "A-ctrl", qls(ret60, U)); print(fmt(row))
r1 = xs_resid(s, size, U)
row, _ = evaluate("A7|sizeneutral", "A", qls(r1, U)); print(fmt(row))
r2 = xs_resid(r1, ret60, U)
row, _ = evaluate("A7|size+mom60neutral", "A", qls(r2, U)); print(fmt(row))
