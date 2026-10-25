"""Deep-dive of POST-HOC candidate A9f 'TOP-TRADER CROWDING CONTRARIAN':
  score_t = -mean(log(top_pos_ls), trailing 14 days); top-50 PIT universe; quintile L/S equal weight,
  $-neutral, gross 1, daily rebalance, flat if < 20 names.  Effective sample 2023-01 -> 2024-12
  (+ fragments Dec-2021, Jun-2022, Dec-2022).
Counted neighbour configs (plateau check, declared before running): S in {1, 7, 30} daily, S14 reb7 = 4.
Diagnostics: funding spanning regression vs funding factors (fund7, fund30 quintile L/S), legs,
concentration, 2023-24-only window, DSR, $200 path."""
from h2lib import *
U = lab.universe("1d", top_k=50)
C = lab.load("close", "1d"); F = lab.load("fund", "1d")
T = lab.load("top_pos_ls", "1d"); T = np.log(T.where(T > 0))
def hold(W, k):
    keep = np.arange(len(W)) % k == 0
    return W.where(pd.Series(keep, index=W.index), np.nan).ffill().fillna(0.0)
s = -T.rolling(14, min_periods=7).mean()
W = qls(s, U)
res = lab.backtest(W, "1d"); m = res["metrics"]; df = res["df"]; r = df["r"]
print("FULL:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
print(lab.yearly(r).round(3).T.to_string())
active = df[df["gross_exposure"] > 0]
print("active days by year:", active.groupby(active.index.year).size().to_dict())
r23 = lab.backtest(W, "1d", start="2023-01-01")
print("2023-24 only:", {k: round(r23['metrics'][k], 3) for k in ['sharpe', 'cagr', 'maxdd', 't_nw', 'turnover_per_year', 'funding_per_year']})
h = r23["df"]["r"]; print("half-year Sharpe:", h.groupby([h.index.year, (h.index.month > 6) + 1]).apply(lambda x: round(x.mean() / x.std() * np.sqrt(365), 2)).to_dict())
q = h.groupby([h.index.year, h.index.quarter]).apply(lambda x: (1 + x).prod() - 1); print("quarterly:", q.round(3).to_dict())
dec = df[["gross_r", "fund_r", "cost_r"]].groupby(df.index.year).sum(); print("decomp:\n", dec.round(3).to_string())
for lag in (1, 2, 3): print(f"lag{lag}", round(lab.backtest(W, "1d", extra_lag=lag)["metrics"]["sharpe"], 2))
print("cost2x", round(lab.backtest(W, "1d", cost=0.0014)["metrics"]["sharpe"], 2), "cost3x", round(lab.backtest(W, "1d", cost=0.0021)["metrics"]["sharpe"], 2))
# neighbours (counted)
for n, Wn in [("A9f_S1", qls(-T, U)), ("A9f_S7", qls(-T.rolling(7, min_periods=4).mean(), U)),
              ("A9f_S30", qls(-T.rolling(30, min_periods=15).mean(), U)), ("A9f_S14_reb7", hold(W, 7))]:
    row, _ = evaluate(n, "A", Wn, n_note="post-hoc neighbour"); print(fmt(row))
# funding spanning (2023-24 window)
fund7 = F.rolling(7, min_periods=5).sum(); fund30 = F.rolling(30, min_periods=15).sum()
f7 = lab.backtest(qls(-fund7, U), "1d", start="2023-01-01")["df"]["r"]
f30 = lab.backtest(qls(-fund30, U), "1d", start="2023-01-01")["df"]["r"]
print("funding factors 2023-24 Sharpe: f7", round(lab.metrics(f7)["sharpe"], 2), " f30", round(lab.metrics(f30)["sharpe"], 2))
X = pd.concat([h, f7, f30], axis=1, keys=["a", "f7", "f30"]).dropna()
print("corr matrix:\n", X.corr().round(2).to_string())
A = np.column_stack([np.ones(len(X)), X["f7"], X["f30"]]); b, *_ = np.linalg.lstsq(A, X["a"].values, rcond=None)
e = X["a"].values - A @ b
alpha_series = X["a"].values - b[1] * X["f7"].values - b[2] * X["f30"].values
print(f"spanning: alpha {b[0]*365:+.1%}/yr  beta_f7 {b[1]:.2f} beta_f30 {b[2]:.2f}  alpha NW-t {lab.newey_west_t(alpha_series, 10):.2f}  resid Sharpe {alpha_series.mean()/alpha_series.std()*np.sqrt(365):.2f}")
# legs vs EW universe
R1 = C.pct_change(fill_method=None).shift(-1)
valid = s.where(U).notna() & R1.notna(); ok = valid.sum(axis=1) >= 20
u = R1.where(valid).mean(axis=1)
Wl = W.clip(lower=0); Ws = -W.clip(upper=0)
lr = (Wl * R1.fillna(0)).sum(axis=1) / Wl.sum(axis=1).replace(0, np.nan)
sr = (Ws * R1.fillna(0)).sum(axis=1) / Ws.sum(axis=1).replace(0, np.nan)
for nm, x in [("long-leg excess", (lr - u)[ok].dropna()), ("short-leg excess", (u - sr)[ok].dropna())]:
    print(f"{nm}: ann {x.mean()*365:+.1%} Sharpe {x.mean()/x.std()*np.sqrt(365):.2f}", (x.groupby(x.index.year).mean()*365).round(3).to_dict())
contrib = (W * R1.fillna(0)).sum().sort_values()
print("top-5:", contrib.tail(5).round(3).to_dict()); print("bottom-5:", contrib.head(5).round(3).to_dict())
print("total", round(contrib.sum(), 3), "top5 share", round(contrib.tail(5).sum() / contrib.sum(), 2))
c = controls().reindex(r.index); print("corr btc", round(r.corr(c["btc"]), 2), " corr xsmom", round(r.corr(c["xsmom14"]), 2))
for n in (44, 73): print(f"DSR n={n}:", round(lab.deflated_sharpe(m["sharpe"], n, m["days"], m["skew"], m["kurt"] + 3), 3),
                        " (2023-24 window:", round(lab.deflated_sharpe(r23["metrics"]["sharpe"], n, r23["metrics"]["days"], r23["metrics"]["skew"], r23["metrics"]["kurt"] + 3), 3), ")")
eq = 200 * (1 + r).cumprod(); print("$200 ->", round(eq.iloc[-1], 1), " min", round(eq.min(), 1))
print("min equity $10/pos:", round(min_equity(W.loc[W.index >= pd.Timestamp('2023-01-01', tz='UTC')]), 0), " avg npos", round(df['n_pos'].mean(), 1))
r.to_frame("r").to_parquet(os.path.join(HERE, "cand_A9f_daily_IS.parquet"))
