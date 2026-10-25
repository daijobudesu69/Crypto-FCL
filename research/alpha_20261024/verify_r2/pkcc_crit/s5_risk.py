"""Step 5 - risk. Pre-registered: worst day/week/month (quintile + 16-pos); worst drawdown episode (dates, legs,
coins driving it, market context); returns through crash windows; behaviour on the 20 worst BTC days and 20 worst
EW-alt days; BTC beta overall / up / down / big-down days and by causal market state (BTC vs 200d SMA; BTC 30d vol
vs its expanding median; 'BTC dominance' proxy = BTC 30d return minus EW-universe 30d return > 0); $200 account sim."""
import numpy as np, pandas as pd
from vlib import *
import account_sim

d = pd.read_parquet(os.path.join(HERE, "base_daily_IS.parquet"))
r = d["r"]; r16 = d["r16"]; rl = d["long_r"]; rs = d["short_r"]
W = pd.read_parquet(os.path.join(HERE, "base_weights_IS.parquet"))
C = lab.load("close", "1d").astype("float64"); R = C.pct_change(fill_method=None)
U = lab.hype_universe("1d", top_k=50)
# market series on the same (realised) index convention as backtest output: return of bar t+1 stamped at t+1 open+1d
m_ = U.astype(float); m_ = m_.div(m_.sum(axis=1), axis=0)
ewr = (m_ * R.shift(-1)).sum(axis=1); ewr.index = ewr.index + pd.Timedelta("1D")
# r[t'] where t' = decision bar + 1D... check alignment: backtest r indexed at bar t+1D realises close[t+1]/close[t]-1,
# i.e. R at row t+1, whose own index (bar open) is t+1 -> shift R index by +1D to match the r index.
btc_al = R["BTCUSDT"].copy()   # R row t+1 (own index) == backtest r index t+1D
J = pd.concat([r.rename("s"), r16.rename("s16"), rl.rename("L"), rs.rename("S"), btc_al.rename("btc"), ewr.rename("ew")], axis=1, sort=True).dropna(subset=["s"])
# sanity: corr(L, ew) should be high
corr_chk = J[["L", "ew", "btc"]].corr().round(2)

def worst(x):
    x = x.dropna()
    wk = (1 + x).rolling(7).apply(np.prod, raw=True) - 1
    mo = (1 + x).groupby(x.index.tz_localize(None).to_period("M")).prod() - 1
    return x.min(), x.idxmin().date(), wk.min(), wk.idxmin().date(), mo.min(), mo.idxmin()

wq = worst(J["s"]); w16 = worst(J["s16"])
eq = (1 + J["s"]).cumprod(); dd = eq / eq.cummax() - 1
trough = dd.idxmin(); peak = eq[:trough].idxmax()
rec = eq[trough:][eq[trough:] >= eq[peak]]
recd = rec.index[0].date() if len(rec) else "not recovered"
seg = J.loc[peak:trough]
# coin attribution during DD: decision days in [peak-1D, trough-1D)
P = (W * R.shift(-1))
P.index = P.index + pd.Timedelta("1D")
Pseg = P.loc[peak + pd.Timedelta("1D"):trough].sum()
worst_coins = Pseg.sort_values().head(6)
# second-worst distinct episode (DD outside the first episode window)
dd2 = dd.copy(); dd2.loc[peak:(rec.index[0] if len(rec) else dd.index[-1])] = 0
tr2 = dd2.idxmin(); pk2_ = eq[:tr2].idxmax()

crash = [("2020-03-10", "2020-03-20", "COVID (book starts 03-10)"), ("2021-05-10", "2021-05-25", "May-2021 crash (05-19)"),
         ("2021-12-03", "2021-12-06", "2021-12-04 flush"), ("2022-05-06", "2022-05-15", "LUNA"),
         ("2022-06-10", "2022-06-20", "3AC/Celsius"), ("2022-11-06", "2022-11-15", "FTX"),
         ("2023-03-09", "2023-03-14", "SVB/USDC"), ("2024-04-12", "2024-04-15", "Apr-2024 flush"),
         ("2024-08-01", "2024-08-07", "yen carry 08-05"), ("2024-12-08", "2024-12-11", "Dec-2024 alt flush")]
cr = []
for a, b, nm in crash:
    x = J.loc[pd.Timestamp(a, tz="UTC"):pd.Timestamp(b, tz="UTC")]
    f = lambda s: (1 + s).prod() - 1
    cr.append((nm, f(x["s"]), f(x["s16"]), f(x["L"]), f(x["S"]), f(x["btc"]), f(x["ew"])))

# tails
wb = J.nsmallest(20, "btc"); we = J.nsmallest(20, "ew")
def beta(y, x):
    z = pd.concat([y, x], axis=1).dropna()
    return np.polyfit(z.iloc[:, 1], z.iloc[:, 0], 1)[0] if len(z) > 20 else np.nan
bt_all = beta(J["s"], J["btc"]); bt_up = beta(J["s"][J["btc"] > 0], J["btc"][J["btc"] > 0])
bt_dn = beta(J["s"][J["btc"] < 0], J["btc"][J["btc"] < 0]); bt_bd = beta(J["s"][J["btc"] < -0.05], J["btc"][J["btc"] < -0.05])
be_all = beta(J["s"], J["ew"])
# causal states, known at decision time (use info up to bar t, applied to r realised at t+1)
cb = C["BTCUSDT"]
sma = cb.rolling(200, min_periods=150).mean()
st_bull = (cb > sma)
vol30 = R["BTCUSDT"].rolling(30).std(); st_hivol = vol30 > vol30.expanding(90).median()
ewc = (m_ * R).sum(axis=1)                       # EW universe daily return at row t
dom = ((1 + R["BTCUSDT"]).rolling(30).apply(np.prod, raw=True) - (1 + ewc).rolling(30).apply(np.prod, raw=True)) > 0
states = {}
for nm, s in (("BTC>200dSMA", st_bull), ("BTC hi-vol", st_hivol), ("BTC beats EW alts 30d (dominance up)", dom)):
    s2 = s.copy(); s2.index = s2.index + pd.Timedelta("1D")
    s2 = s2.reindex(J.index)
    for val in (True, False):
        mk = s2 == val
        x = J[mk]
        states[(nm, val)] = (int(mk.sum()), sharpe(x["s"]), x["s"].mean() * 365, beta(x["s"], x["btc"]), sharpe(x["s16"]))

# $200 account sim
sims = {}
for nm, Wx in (("quintile", W), ("16pos", qls(-pkcc(30), U, n=8))):
    out, sm = account_sim.simulate(Wx, "1d", equity0=200.0, min_order=10.0, cost=COST)
    e = out["equity"]; dr = e.pct_change().dropna()
    sims[nm] = (sm["equity_end"], sm["maxdd"], sm["skipped_below_min"], sm["orders"], sharpe(dr), sm["fees_usd"], e.min())

L = ["## Step 5 - risk",
     f"- alignment sanity corr(long leg, EW universe, BTC): L~EW {corr_chk.loc['L', 'ew']}, L~BTC {corr_chk.loc['L', 'btc']}",
     f"- Worst day / 7d / month, quintile: {wq[0]:+.1%} ({wq[1]}), {wq[2]:+.1%} (7d to {wq[3]}), {wq[4]:+.1%} ({wq[5]}). "
     f"16-pos: {w16[0]:+.1%} ({w16[1]}), {w16[2]:+.1%} ({w16[3]}), {w16[4]:+.1%} ({w16[5]}).",
     f"- Worst drawdown (quintile): {dd.min():.1%}, peak {peak.date()} -> trough {trough.date()} ({(trough - peak).days} d), recovered {recd}. "
     f"Over the episode: long leg {(1 + seg['L']).prod() - 1:+.1%}, short leg {(1 + seg['S']).prod() - 1:+.1%}, "
     f"BTC {(1 + seg['btc']).prod() - 1:+.1%}, EW universe {(1 + seg['ew']).prod() - 1:+.1%}. "
     f"Worst coin contributions (price PnL, fraction of equity): " + ", ".join(f"{k} {v:+.1%}" for k, v in worst_coins.items()),
     f"- 2nd-worst distinct episode: {dd2.min():.1%}, {pk2_.date()} -> {tr2.date()}.",
     "", "| crash window | quintile | 16-pos | long leg | short leg | BTC | EW univ |", "|---|---|---|---|---|---|---|"]
for x in cr:
    L.append(f"| {x[0]} | {x[1]:+.1%} | {x[2]:+.1%} | {x[3]:+.1%} | {x[4]:+.1%} | {x[5]:+.1%} | {x[6]:+.1%} |")
L += ["",
      f"- 20 worst BTC days (mean BTC {wb['btc'].mean():+.1%}): strategy mean {wb['s'].mean():+.2%}, hit rate {(wb['s'] > 0).mean():.0%}, worst {wb['s'].min():+.1%}. "
      f"20 worst EW-alt days (mean {we['ew'].mean():+.1%}): strategy mean {we['s'].mean():+.2%}, hit {(we['s'] > 0).mean():.0%}, worst {we['s'].min():+.1%}.",
      f"- BTC beta: all {bt_all:+.3f}, BTC-up days {bt_up:+.3f}, BTC-down days {bt_dn:+.3f}, BTC < -5% days {bt_bd:+.3f}; beta to EW universe {be_all:+.3f}.",
      "", "| causal state | days | Sharpe | ann. mean | BTC beta | 16-pos Sharpe |", "|---|---|---|---|---|---|"]
for (nm, val), x in states.items():
    L.append(f"| {nm} = {val} | {x[0]} | {x[1]:.2f} | {x[2]:+.1%} | {x[3]:+.3f} | {x[4]:.2f} |")
L += ["", "- $200 account sim (account_sim.py, $10 min order, HYPE cost + funding, 2020-03..2024-12):"]
for nm, x in sims.items():
    L.append(f"  * {nm}: end equity ${x[0]:,.0f}, MaxDD {x[1]:.1%}, min equity ${x[6]:.0f}, Sharpe {x[4]:.2f}, orders {x[3]}, "
             f"positions skipped below $10: {x[2]}, fees ${x[5]:,.0f}")
append("\n".join(L))
print("\n".join(L))
