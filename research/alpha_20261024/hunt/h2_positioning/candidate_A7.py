"""Deep-dive of the family-A candidate (spec frozen from the plateau centre of family_A7_refine.py):
  RETAIL-CROWDING CONTRARIAN (A7_S14_reb1):
  score_t = -mean(log(global_ls), trailing 14 days)   [low retail long/short ACCOUNT ratio = long]
  universe = lab.universe('1d', top_k=50) (PIT 30d median quote-volume top-50, >=60d history),
  quintile L/S equal-weight, $-neutral, gross 1x, daily rebalance at 00:00 UTC close, flat if <20 names.
Extra counted configs here: momentum-neutral version (resid on 60d return) = 1.
Diagnostics only: leg excess vs EW universe, coin concentration, half-years, lag 2, funding-factor corr."""
from h2lib import *
U = lab.universe("1d", top_k=50)
C = lab.load("close", "1d"); F = lab.load("fund", "1d")
G = lab.load("global_ls", "1d"); G = G.where(G > 0)
s = -np.log(G).rolling(14, min_periods=7).mean()
W = qls(s, U)
row, res = evaluate("CAND_A7_S14_reb1", "A", W); print(fmt(row))
m = res["metrics"]; df = res["df"]; r = df["r"]
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
print(lab.yearly(r).round(3).T.to_string())
# half-year sharpe
h = r.groupby([r.index.year, (r.index.month > 6).astype(int) + 1]).apply(lambda x: x.mean() / x.std() * np.sqrt(365))
print("half-year Sharpe:", h.round(2).to_dict())
for lag in (1, 2, 3):
    print(f"lag{lag} Sharpe", round(lab.backtest(W, "1d", extra_lag=lag)["metrics"]["sharpe"], 2))
print("cost 2x", round(lab.backtest(W, "1d", cost=0.0014)["metrics"]["sharpe"], 2), " cost 3x", round(lab.backtest(W, "1d", cost=0.0021)["metrics"]["sharpe"], 2))
print("no funding", round(lab.backtest(W, "1d", funding=False)["metrics"]["sharpe"], 2))
# PnL decomposition per year
dec = df[["gross_r", "fund_r", "cost_r"]].groupby(df.index.year).sum()
print("PnL decomposition (sum of daily, per year):\n", dec.round(3).to_string())
# leg excess vs EW universe (gross, next-day)
R1 = C.pct_change(fill_method=None).shift(-1)
valid = s.where(U).notna() & R1.notna()
ok = valid.sum(axis=1) >= 20
u = R1.where(valid).mean(axis=1)
Wl = W.clip(lower=0); Ws = -W.clip(upper=0)
lr = (Wl * R1.fillna(0)).sum(axis=1) / Wl.sum(axis=1).replace(0, np.nan)
sr = (Ws * R1.fillna(0)).sum(axis=1) / Ws.sum(axis=1).replace(0, np.nan)
ex_l = (lr - u)[ok].dropna(); ex_s = (u - sr)[ok].dropna()
for nm, x in [("long-leg excess vs EW univ", ex_l), ("short-leg excess (univ - shorts)", ex_s)]:
    print(f"{nm}: ann {x.mean()*365:+.1%}  Sharpe {x.mean()/x.std()*np.sqrt(365):.2f}  by year", (x.groupby(x.index.year).mean()*365).round(3).to_dict())
# coin concentration
contrib = (W * R1.fillna(0)).sum().sort_values()
print("top-5 contributors:", contrib.tail(5).round(3).to_dict())
print("bottom-5 contributors:", contrib.head(5).round(3).to_dict())
print("total gross", round(contrib.sum(), 3), " share of top-5:", round(contrib.tail(5).sum() / contrib.sum(), 2))
print("ex-top5 coins gross sum:", round(contrib.sum() - contrib.tail(5).sum(), 3))
# controls
fund7 = F.rolling(7, min_periods=5).sum()
rf = lab.backtest(qls(-fund7, U), "1d")["df"]["r"].reindex(r.index)
print("corr with funding factor:", round(r.corr(rf), 2))
# deflated sharpe
T = m["days"]
for n in (36, 70):
    print(f"DSR n_trials={n}:", round(lab.deflated_sharpe(m["sharpe"], n, T, m["skew"], m["kurt"] + 3), 3))
# $200 equity path
eq = 200 * (1 + r).cumprod()
print("equity from $200: final", round(eq.iloc[-1], 1), " min", round(eq.min(), 1), " maxDD", round(m["maxdd"], 3))
print("avg names per leg", round(df["n_pos"].mean() / 2, 1), " min equity for $10/position:", round(min_equity(W.loc[W.index >= pd.Timestamp(m['start'], tz='UTC')]), 0))
r.to_frame("r").to_parquet(os.path.join(HERE, "cand_A7_daily_IS.parquet"))
# counted extra: momentum-neutral
ret60 = C / C.shift(60) - 1
row, _ = evaluate("CAND_A7_S14|mom60neutral", "A", qls(xs_resid(s, ret60, U), U)); print(fmt(row))
