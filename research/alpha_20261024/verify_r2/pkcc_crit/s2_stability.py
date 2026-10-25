"""Step 2 - stability. Pre-registered: rolling 365d Sharpe path (quintile, 16-pos, lag1); leave-one-year-out;
ex-2021; 2023-24 alone; each year alone; Sharpe ex 10 best days (and ex 10 best + 10 worst); top-3 months removed;
leave-one-quarter-out min; leave-one-coin-out (re-ranked universe without the coin) for every coin ever held.
DISP7-specific episode items: N/A for PKCC30 (no regime state); replaced by the month/quarter/coin removals."""
import numpy as np, pandas as pd
from vlib import *

d = pd.read_parquet(os.path.join(HERE, "base_daily_IS.parquet"))
r = d["r"]; r16 = d["r16"]
W = pd.read_parquet(os.path.join(HERE, "base_weights_IS.parquet"))
rl1 = bt(W, lag=1)["df"]["r"]

def roll(x):
    return (x.rolling(365).mean() / x.rolling(365).std() * np.sqrt(365)).dropna()

rows = []
for nm, x in (("quintile", r), ("16pos", r16), ("quintile_lag1", rl1)):
    rs = roll(x)
    rows.append((nm, rs.min(), rs.idxmin().date(), rs.quantile(.1), rs.median(), (rs > 0).mean(), (rs > 0.5).mean()))
rr = pd.DataFrame(rows, columns=["book", "min", "min_date", "p10", "median", "frac>0", "frac>0.5"])
pd.DataFrame({"quintile": roll(r), "16pos": roll(r16), "lag1": roll(rl1)}).to_csv(os.path.join(HERE, "rolling365.csv"))
# quarterly path of rolling Sharpe (end of quarter)
rq = roll(r).resample("QE").last()

yrs = sorted(set(r.index.year))
loyo = {y: sharpe(r[r.index.year != y]) for y in yrs}
alone = {y: sharpe(r[r.index.year == y]) for y in yrs}
alone16 = {y: sharpe(r16[r16.index.year == y]) for y in yrs}
ex21 = sharpe(r[r.index.year != 2021])
s2324 = sharpe(win(r, "2023-01-01", "2025-01-01")); s2324_16 = sharpe(win(r16, "2023-01-01", "2025-01-01"))
def ex_best(x, k=10, worst=False):
    x = x.dropna(); drop = x.nlargest(k).index
    if worst:
        drop = drop.union(x.nsmallest(k).index)
    return sharpe(x.drop(drop))
eb = {nm: (ex_best(x), ex_best(x, worst=True)) for nm, x in (("quintile", r), ("16pos", r16), ("lag1", rl1))}
mo = (1 + r).groupby(r.index.to_period("M") if r.index.tz is None else r.index.tz_localize(None).to_period("M")).prod() - 1
top3 = mo.nlargest(3)
ri = r.copy(); per = ri.index.tz_localize(None).to_period("M")
ex_top3m = sharpe(ri[~per.isin(top3.index)])
qq = ri.index.tz_localize(None).to_period("Q")
loqo = {str(q): sharpe(ri[qq != q]) for q in qq.unique()}
loqo_s = pd.Series(loqo)
best_day_share = r.nlargest(10).sum() / r.sum()
# PnL share of the top 5% days
top5pct = r.nlargest(int(0.05 * len(r))).sum() / r.sum()

# leave-one-coin-out
pk2, cc2 = comps()
sig = -pkcc(30, pk2=pk2, cc2=cc2)
U = lab.hype_universe("1d", top_k=50)
held = W.columns[(W != 0).any()]
looc = {}
for coin in held:
    Ux = U.copy(); Ux[coin] = False
    Wx = qls(sig, Ux)
    looc[coin] = bt(Wx)["metrics"]["sharpe"]
looc = pd.Series(looc).sort_values()
# drop top-k PnL coins jointly (price PnL attribution from base weights)
R = lab.load("close", "1d").astype("float64").pct_change(fill_method=None)
pnl = (W * R.shift(-1)).sum().sort_values(ascending=False)
joint = {}
for k in (3, 5, 10):
    Ux = U.copy(); Ux[list(pnl.index[:k])] = False
    joint[k] = bt(qls(sig, Ux))["metrics"]["sharpe"]
pd.DataFrame({"looc_sharpe": looc}).to_csv(os.path.join(HERE, "leave_one_coin_out.csv"))

L = ["## Step 2 - stability",
     "Rolling 365d Sharpe:", "", "| book | min | min date | p10 | median | frac>0 | frac>0.5 |", "|---|---|---|---|---|---|---|"]
for _, x in rr.iterrows():
    L.append(f"| {x['book']} | {x['min']:.2f} | {x['min_date']} | {x['p10']:.2f} | {x['median']:.2f} | {x['frac>0']:.0%} | {x['frac>0.5']:.0%} |")
L += ["", "- Rolling 365d Sharpe at quarter ends (quintile): " + ", ".join(f"{i.year}Q{i.quarter} {v:.2f}" for i, v in rq.items()),
      "- Each year alone (quintile): " + ", ".join(f"{y} {v:.2f}" for y, v in alone.items()),
      "- Each year alone (16-pos): " + ", ".join(f"{y} {v:.2f}" for y, v in alone16.items()),
      "- Leave-one-year-out (quintile): " + ", ".join(f"ex{y} {v:.2f}" for y, v in loyo.items()),
      f"- ex-2021 {ex21:.2f}; 2023-24 alone {s2324:.2f} (16-pos {s2324_16:.2f}); 2023 alone {alone[2023]:.2f}",
      f"- Sharpe ex 10 best days: quintile {eb['quintile'][0]:.2f} (ex 10 best + 10 worst {eb['quintile'][1]:.2f}); "
      f"16-pos {eb['16pos'][0]:.2f} ({eb['16pos'][1]:.2f}); lag1 {eb['lag1'][0]:.2f} ({eb['lag1'][1]:.2f}). "
      f"10 best days = {best_day_share:.0%} of total summed net return; top 5% days = {top5pct:.0%}.",
      f"- Top-3 months {', '.join(f'{p} {v:+.1%}' for p, v in top3.items())} removed -> Sharpe {ex_top3m:.2f}",
      f"- Leave-one-quarter-out: min {loqo_s.min():.2f} (ex {loqo_s.idxmin()}), max {loqo_s.max():.2f} (ex {loqo_s.idxmax()})",
      f"- Leave-one-coin-out ({len(looc)} coins ever held, universe re-ranked): min {looc.min():.2f} (ex {looc.idxmin()}), "
      f"p10 {looc.quantile(.1):.2f}, median {looc.median():.2f}, max {looc.max():.2f} (ex {looc.idxmax()}). Lowest 5: "
      + ", ".join(f"{k} {v:.2f}" for k, v in looc.head(5).items()),
      f"- Joint removal of top-k price-PnL coins ({', '.join(pnl.index[:5])} ...): top3 {joint[3]:.2f}, top5 {joint[5]:.2f}, top10 {joint[10]:.2f}",
      "- DISP7 regime-episode items: not applicable (PKCC30 has no state variable); month/quarter/coin removals used instead."]
append("\n".join(L))
print("\n".join(L))
