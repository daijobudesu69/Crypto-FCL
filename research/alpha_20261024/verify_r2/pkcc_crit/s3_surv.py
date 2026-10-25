"""Step 3 - survivorship. Pre-registered variants (same signal, same quintile book, HYPE cost):
 base   HYPE top-50
 a      HYPE top-50 restricted to coins whose first bar is before 2022-01-01 (no post-2021 listings)
 b1     legacy majors: first bar before 2020-03-01 (15 coins: BTC ETH BCH XRP LTC TRX ETC LINK XLM ADA XMR DASH ZEC ATOM BNB)
 b2     coins listed before 2021-01-01 (27)
 c      HYPE top-50 with min_hist_days = 180 (no coin in its first 180 days)
 d      Binance top-50 (all 150 survivors, not HYPE-filtered)  and  e  Binance top-50 restricted to NON-HYPE coins
For each: Sharpe, sub-periods, lag1, long-leg excess vs EW (same universe), EW-minus-short excess, raw legs.
Crash diagnostics: leg membership of coins in a crash (trailing 5d return <= -25%) and their next-day return;
holdings during LUNA (2022-05-07..13) and FTX (2022-11-06..14) weeks."""
import numpy as np, pandas as pd
from vlib import *

C = lab.load("close", "1d").astype("float64")
R = C.pct_change(fill_method=None)
first = C.apply(lambda s: s.first_valid_index())
pk2, cc2 = comps()
sig = -pkcc(30, pk2=pk2, cc2=cc2)
UH = lab.hype_universe("1d", top_k=50)
UB = lab.universe("1d", top_k=50)
hs = set(lab.hype_symbols())

def keep(U, cols):
    m = pd.Series(U.columns.isin(list(cols)), index=U.columns)
    return U & m

pre22 = first[first < pd.Timestamp("2022-01-01", tz="UTC")].index
pre2003 = first[first < pd.Timestamp("2020-03-01", tz="UTC")].index
pre21 = first[first < pd.Timestamp("2021-01-01", tz="UTC")].index
var = {
    "base HYPE top50": UH,
    "a  listed<2022": keep(UH, pre22),
    "b1 legacy<2020-03 (15)": keep(lab.hype_universe("1d"), pre2003),
    "b2 listed<2021 (27)": keep(lab.hype_universe("1d"), pre21),
    "c  min_hist 180d": lab.hype_universe("1d", top_k=50, min_hist_days=180),
    "d  Binance top50": UB,
    "e  Binance top50 non-HYPE": keep(UB, [x for x in C.columns if x not in hs]),
}

def ew(U):
    m = U.astype(float); return m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

def legs(W, U):
    WL = W.clip(lower=0); WS = W.clip(upper=0)
    gl = WL.sum(axis=1).replace(0, np.nan); gs = (-WS).sum(axis=1).replace(0, np.nan)
    rl = bt(WL.div(gl, axis=0).fillna(0))["df"]["r"]; rs = bt((-WS).div(gs, axis=0).fillna(0))["df"]["r"]
    re = bt(ew(U))["df"]["r"]
    j = pd.concat([rl.rename("l"), rs.rename("s"), re.rename("e")], axis=1).dropna()
    return j

age = C.notna().cumsum()
rows = []
for nm, U in var.items():
    W = qls(sig, U)
    res = bt(W); m = res["metrics"]; r = res["df"]["r"]
    l1 = bt(W, lag=1)["metrics"]["sharpe"]
    j = legs(W, U)
    sb = subs(r)
    al = age.where(W > 0).stack().median(); as_ = age.where(W < 0).stack().median()
    rows.append({"variant": nm, "sharpe": m["sharpe"], "2020-21": sb["2020-21"], "2022": sb["2022"], "2023-24": sb["2023-24"],
                 "lag1": l1, "maxdd": m["maxdd"], "npos": m["avg_n_pos"],
                 "L-EW xsSR": sharpe(j["l"] - j["e"]), "EW-S xsSR": sharpe(j["e"] - j["s"]),
                 "L-EW ann": (j["l"] - j["e"]).mean() * 365, "EW-S ann": (j["e"] - j["s"]).mean() * 365,
                 "age_long_med": al, "age_short_med": as_})
T = pd.DataFrame(rows).set_index("variant")
print(T.round(2).to_string())

# crash diagnostics on base
W = qls(sig, UH)
r5 = C / C.shift(5) - 1
crash = (r5 <= -0.25) & UH
inL = (W > 0) & crash; inS = (W < 0) & crash; inN = (W == 0) & crash
nxt = R.shift(-1)
cd = {k: (int(v.sum().sum()), float(nxt.where(v).stack().mean())) for k, v in (("long", inL), ("short", inS), ("none", inN))}
# big subsequent losers: coin-days held long whose next 10d return <= -40%
f10 = C.shift(-10) / C - 1
bigL = ((W > 0) & (f10 <= -0.40)).sum().sum(); bigS = ((W < 0) & (f10 <= -0.40)).sum().sum()
# leg membership in crash weeks
def held(a, b):
    w = W[(W.index >= pd.Timestamp(a, tz="UTC")) & (W.index < pd.Timestamp(b, tz="UTC"))]
    return {"long": sorted(set(w.columns[(w > 0).any()])), "short": sorted(set(w.columns[(w < 0).any()]))}
luna = held("2022-05-07", "2022-05-14"); ftx = held("2022-11-06", "2022-11-15")
# SOL in FTX week
sol = W.loc[(W.index >= pd.Timestamp("2022-11-05", tz="UTC")) & (W.index < pd.Timestamp("2022-11-16", tz="UTC")), "SOLUSDT"]
solr = R.loc[sol.index, "SOLUSDT"]
L = ["## Step 3 - survivorship",
     "Universe = 150 Binance perps alive Sep 2026, further filtered to the 92 listed on HYPE TODAY (both are future information).",
     "", "| variant | Sharpe | 2020-21 | 2022 | 2023-24 | lag1 | MaxDD | npos | long-EW xs SR | EW-short xs SR | long-EW %/yr | EW-short %/yr | med age long (d) | med age short (d) |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for nm, x in T.iterrows():
    L.append(f"| {nm} | {x['sharpe']:.2f} | {x['2020-21']:.2f} | {x['2022']:.2f} | {x['2023-24']:.2f} | {x['lag1']:.2f} | {x['maxdd']:.0%} | "
             f"{x['npos']:.1f} | {x['L-EW xsSR']:.2f} | {x['EW-S xsSR']:.2f} | {x['L-EW ann']:+.1%} | {x['EW-S ann']:+.1%} | {x['age_long_med']:.0f} | {x['age_short_med']:.0f} |")
L += ["",
      f"- Crash coin-days (trailing 5d <= -25%, in universe): held LONG {cd['long'][0]} (next-day mean {cd['long'][1]:+.2%}), "
      f"SHORT {cd['short'][0]} ({cd['short'][1]:+.2%}), not held {cd['none'][0]} ({cd['none'][1]:+.2%}).",
      f"- Coin-days held whose next 10d return <= -40%: long {int(bigL)}, short {int(bigS)}.",
      f"- LUNA week 2022-05-07..13 holdings: long {luna['long']}; short {luna['short']}",
      f"- FTX week 2022-11-06..14 holdings: long {ftx['long']}; short {ftx['short']}",
      "- SOL during FTX (weight, next-day ret): " + ", ".join(f"{i.date()} {w:+.3f}/{rr:+.1%}" for i, w, rr in zip(sol.index, sol.values, R.loc[sol.index, 'SOLUSDT'].shift(-1).values))]
append("\n".join(L))
print("\n".join(L))
T.to_csv(os.path.join(HERE, "survivorship.csv"))
