"""Diagnostic only (ex-post, not a tradable config): concentration of A7_S14_reb1. Remove the top-k
contributing coins from the universe and re-run; leave-one-coin-out min Sharpe over top-15 contributors."""
from h2lib import *
U = lab.universe("1d", top_k=50)
C = lab.load("close", "1d"); G = lab.load("global_ls", "1d"); G = G.where(G > 0)
s = -np.log(G).rolling(14, min_periods=7).mean()
R1 = C.pct_change(fill_method=None).shift(-1)
W = qls(s, U)
contrib = (W * R1.fillna(0)).sum().sort_values(ascending=False)
for k in (1, 3, 5, 10):
    drop = list(contrib.index[:k]); U2 = U.copy(); U2[drop] = False
    m = lab.backtest(qls(s, U2), "1d")["metrics"]
    print(f"drop top-{k} contributors {drop[:5]}...: Sharpe {m['sharpe']:.2f} CAGR {m['cagr']:+.1%}")
shs = {}
for c in contrib.index[:15]:
    U2 = U.copy(); U2[c] = False
    shs[c] = round(lab.backtest(qls(s, U2), "1d")["metrics"]["sharpe"], 2)
print("leave-one-out Sharpe:", shs)
r = lab.backtest(W, "1d")["df"]["r"]
q = r.groupby([r.index.year, r.index.quarter]).apply(lambda x: (1 + x).prod() - 1)
print("quarterly returns:", q.round(3).to_dict())
