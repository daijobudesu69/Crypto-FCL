"""A7 refinement (pre-registered BEFORE running): A7 = -log(global_ls) level is a weekly-horizon
signal (IC1 ~0, IC7 +0.029) traded daily on noisy daily ratios -> turnover. Grid:
  smoothing S in {7, 14} (trailing mean of log global_ls)  x  rebalance every {1, 3, 7} days   6
  + raw (S=1) with rebalance every 7                                                          1
  + top-100 universe for the S=7/reb=7 cell (chosen a priori as the 'natural weekly' cell)    1
TOTAL 8. Universe top-50 PIT, quintile L/S, cost 0.07% + funding."""
from h2lib import *
U50 = lab.universe("1d", top_k=50); U100 = lab.universe("1d", top_k=100)
G = lab.load("global_ls", "1d"); G = G.where(G > 0); lg = np.log(G)
def hold(W, k):
    if k == 1: return W
    keep = np.arange(len(W)) % k == 0
    return W.where(pd.Series(keep, index=W.index), np.nan).ffill().fillna(0.0)
cells = [(S, k) for S in (7, 14) for k in (1, 3, 7)] + [(1, 7)]
for S, k in cells:
    s = -(lg.rolling(S, min_periods=max(1, S // 2)).mean() if S > 1 else lg)
    row, res = evaluate(f"A7_S{S}_reb{k}", "A", hold(qls(s, U50), k)); print(fmt(row))
s = -lg.rolling(7, min_periods=4).mean()
row, res = evaluate("A7_S7_reb7|top100", "A", hold(qls(s, U100), 7)); print(fmt(row))
