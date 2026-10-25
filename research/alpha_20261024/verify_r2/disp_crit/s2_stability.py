"""Step 2: stability. Rolling 365d Sharpe, leave-one-year-out, ex-2021, 2023-24 alone, ex 10 best days,
regime episodes (state spells): count, per-episode P&L, leave-one-episode-out, top-k episodes removed."""
import numpy as np, pandas as pd
import dc
d = pd.read_csv(dc.os.path.join(dc.HERE, "disp7_terc_daily.csv"), index_col=0, parse_dates=True)
r = d["r"]; s = d["state"]
L = ["## Step 2 — stability"]
roll = r.rolling(365).apply(lambda x: x.mean() / x.std() * np.sqrt(365), raw=True).dropna()
q = roll.resample("QE").last()
L.append(f"Rolling 365d Sharpe ({len(roll)} obs): min {roll.min():.2f} ({roll.idxmin().date()}), max {roll.max():.2f} ({roll.idxmax().date()}), "
         f"median {roll.median():.2f}; share of windows > 0: {(roll > 0).mean():.0%}, > 0.5: {(roll > 0.5).mean():.0%}, > 1.0: {(roll > 1).mean():.0%}.")
L.append("Quarter-end values: " + ", ".join(f"{i.strftime('%Y-%m')} {v:.2f}" for i, v in q.items()))
roll.to_csv(dc.os.path.join(dc.HERE, "rolling365_sharpe.csv"))
yrs = sorted(set(r.index.year))
L.append("Leave-one-year-out Sharpe: " + ", ".join(f"ex{y} {dc.sharpe(r[r.index.year != y]):.2f}" for y in yrs))
L.append(f"ex-2021: {dc.sharpe(r[r.index.year != 2021]):.2f} (CAGR {((1 + r[r.index.year != 2021]).prod() ** (365 / (r.index.year != 2021).sum()) - 1):+.1%}); "
         f"2023-24 alone {dc.sharpe(dc.between(r, '2023-01-01', '2025-01-01')):.2f}; 2024 alone {dc.sharpe(r[r.index.year == 2024]):.2f}; "
         f"2022-24 {dc.sharpe(dc.between(r, '2022-01-01', '2025-01-01')):.2f}; 2020-H2 {dc.sharpe(r[r.index.year == 2020]):.2f}; "
         f"last 18 months (2023-07..2024-12) {dc.sharpe(dc.between(r, '2023-07-01', '2025-01-01')):.2f}.")
srt = r.sort_values()
L.append(f"Ex 10 best days: {dc.sharpe(srt.iloc[:-10]):.2f}; ex 20 best {dc.sharpe(srt.iloc[:-20]):.2f}; ex 10 best & 10 worst {dc.sharpe(srt.iloc[10:-10]):.2f}. "
         f"10 best days sum {srt.iloc[-10:].sum():+.1%} vs total arithmetic {r.sum():+.1%}; dates: " + ", ".join(str(i.date()) for i in srt.index[-10:]))
# episodes
sp = (s != s.shift()).cumsum()
ep = pd.DataFrame({"start": r.index.to_series().groupby(sp).first(), "end": r.index.to_series().groupby(sp).last(),
                   "state": s.groupby(sp).first(), "days": r.groupby(sp).size(),
                   "logpnl": np.log1p(r).groupby(sp).sum()})
ep.to_csv(dc.os.path.join(dc.HERE, "episodes.csv"))
n = len(ep); win = (ep.logpnl > 0).mean()
tep = ep.logpnl.mean() / ep.logpnl.std() * np.sqrt(n)
L.append(f"Regime episodes (contiguous state spells): {n} total ({(ep.state == 1).sum()} LONG-SPREAD, {(ep.state == -1).sum()} SHORT-SPREAD); "
         f"median length {ep.days.median():.0f} d, mean {ep.days.mean():.0f} d, longest {ep.days.max()} d. Episodes with P&L > 0: {win:.0%}. "
         f"t-stat of mean episode log-P&L across episodes: {tep:.2f} (n={n}).")
for sv in (1, -1):
    e = ep[ep.state == sv]
    L.append(f"  state {sv:+d}: {len(e)} episodes, win {(e.logpnl > 0).mean():.0%}, mean {e.logpnl.mean():+.1%}, median {e.logpnl.median():+.1%}, t {e.logpnl.mean() / e.logpnl.std() * np.sqrt(len(e)):.2f}")
top = ep.sort_values("logpnl", ascending=False)
tot = ep.logpnl.sum()
L.append(f"Total log P&L {tot:+.2f}. Top-3 episodes = {top.logpnl.iloc[:3].sum() / tot:.0%} of it, top-5 = {top.logpnl.iloc[:5].sum() / tot:.0%}, top-10 = {top.logpnl.iloc[:10].sum() / tot:.0%}.")
L.append("Top-5 episodes: " + "; ".join(f"{a.start.date()}..{a.end.date()} st {a.state:+.0f} {a.days}d {a.logpnl:+.1%}" for a in top.head(5).itertuples()))
L.append("Worst-5 episodes: " + "; ".join(f"{a.start.date()}..{a.end.date()} st {a.state:+.0f} {a.days}d {a.logpnl:+.1%}" for a in top.tail(5).itertuples()))
loo = []
for k in ep.index:
    loo.append(dc.sharpe(r[sp != k]))
loo = np.array(loo)
L.append(f"Leave-one-episode-out Sharpe: min {loo.min():.2f}, median {np.median(loo):.2f}, max {loo.max():.2f}.")
for k in (1, 3, 5, 10):
    drop = top.index[:k]
    L.append(f"  top-{k} episodes removed (days dropped): Sharpe {dc.sharpe(r[~sp.isin(drop)]):.2f}; set flat instead: {dc.sharpe(r.where(~sp.isin(drop), 0.0)):.2f}")
dc.append("\n".join(L))
