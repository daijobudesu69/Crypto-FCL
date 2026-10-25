"""Final summary: trial counts from runlog.csv, candidate stats, and one counted combo
(50/50 weight blend of A7_S14 and A9f; both 'crowded positioning is contrarian')."""
from h2lib import *
U = lab.universe("1d", top_k=50)
G = lab.load("global_ls", "1d"); G = np.log(G.where(G > 0)); T = lab.load("top_pos_ls", "1d"); T = np.log(T.where(T > 0))
WA7 = qls(-G.rolling(14, min_periods=7).mean(), U); WA9 = qls(-T.rolling(14, min_periods=7).mean(), U)
Wc = 0.5 * WA7 + 0.5 * WA9.reindex_like(WA7).fillna(0)
row, res = evaluate("COMBO_A7+A9f_50/50", "A", Wc, n_note="post-hoc blend"); print(fmt(row))
print("combo 2023-24:", round(lab.backtest(Wc, "1d", start="2023-01-01")["metrics"]["sharpe"], 2))
log = pd.read_csv(LOG)
log = log.drop_duplicates("name", keep="last")
print(log.groupby("family").size().to_dict())
for n, W in [("A7_S14", WA7), ("A9f", WA9), ("combo", Wc)]:
    r = lab.backtest(W, "1d")
    y = lab.yearly(r["df"]["r"])
    print(n, "yearly ret:", y["ret"].round(3).to_dict(), "sharpe:", y["sharpe"].round(2).to_dict(), "min_eq", round(min_equity(W.loc[W.index >= pd.Timestamp('2023-01-01', tz='UTC')]), 0))
nA = int((log.family == "A").sum()); nB = int((log.family == "B").sum())
for n, W, st in [("A7_S14", WA7, None), ("A9f", WA9, None), ("A9f 2023-24", WA9, "2023-01-01"), ("combo", Wc, None)]:
    m = lab.backtest(W, "1d", start=st)["metrics"]
    print(n, "DSR famA n=", nA, round(lab.deflated_sharpe(m["sharpe"], nA, m["days"], m["skew"], m["kurt"] + 3), 3),
          " all n=", nA + nB, round(lab.deflated_sharpe(m["sharpe"], nA + nB, m["days"], m["skew"], m["kurt"] + 3), 3))
