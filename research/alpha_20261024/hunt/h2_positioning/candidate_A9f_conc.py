"""Diagnostic only (ex-post): A9f concentration (drop top-k contributors, leave-one-out), 2024Q4 dependence,
correlation with A7, and top-100 universe check (counted: 1)."""
from h2lib import *
U = lab.universe("1d", top_k=50)
C = lab.load("close", "1d"); T = lab.load("top_pos_ls", "1d"); T = np.log(T.where(T > 0))
G = lab.load("global_ls", "1d"); G = np.log(G.where(G > 0))
s = -T.rolling(14, min_periods=7).mean()
R1 = C.pct_change(fill_method=None).shift(-1)
W = qls(s, U)
contrib = (W * R1.fillna(0)).sum().sort_values(ascending=False)
for k in (1, 3, 5, 10):
    drop = list(contrib.index[:k]); U2 = U.copy(); U2[drop] = False
    m = lab.backtest(qls(s, U2), "1d", start="2023-01-01")["metrics"]
    print(f"2023-24 drop top-{k} {drop[:5]}: Sharpe {m['sharpe']:.2f} CAGR {m['cagr']:+.1%}")
loo = {}
for c in contrib.index[:12]:
    U2 = U.copy(); U2[c] = False
    loo[c] = round(lab.backtest(qls(s, U2), "1d", start="2023-01-01")["metrics"]["sharpe"], 2)
print("leave-one-out 2023-24 Sharpe:", loo)
r = lab.backtest(W, "1d", start="2023-01-01")["df"]["r"]
x = r[r.index < pd.Timestamp("2024-10-01", tz="UTC")]
print("2023-01..2024-09 (ex 2024Q4) Sharpe:", round(x.mean() / x.std() * np.sqrt(365), 2), " CAGR", round((1 + x).prod() ** (365 / len(x)) - 1, 3))
q4 = (W * R1.fillna(0))[(W.index >= pd.Timestamp("2024-10-01", tz="UTC"))].sum().sort_values()
print("2024Q4 top contributors:", q4.tail(5).round(3).to_dict(), " worst:", q4.head(3).round(3).to_dict())
rA7 = lab.backtest(qls(-G.rolling(14, min_periods=7).mean(), U), "1d", start="2023-01-01")["df"]["r"]
print("corr(A9f, A7) 2023-24:", round(r.corr(rA7.reindex(r.index)), 2))
row, _ = evaluate("A9f|top100", "A", qls(s, lab.universe("1d", top_k=100)), n_note="post-hoc robustness"); print(fmt(row))
