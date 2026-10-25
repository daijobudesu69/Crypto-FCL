"""Step 3b - survivorship follow-ups (diagnostics, decided after seeing s3: crash days are held long 2x as often).
 1. price-PnL attribution of base book by coin age bucket (<180d, 180-365d, >365d) and by crash state, per leg
 2. Sharpe when crash coin-days are excluded from the universe (trailing 5d <= -25%), re-ranked
 3. non-HYPE survivors ranked among themselves (top-30 by liquidity, min hist 60d) -> is the edge HYPE-list specific?
 4. SOL close around FTX (data sanity)"""
import numpy as np, pandas as pd
from vlib import *

C = lab.load("close", "1d").astype("float64"); R = C.pct_change(fill_method=None)
pk2, cc2 = comps(); sig = -pkcc(30, pk2=pk2, cc2=cc2)
UH = lab.hype_universe("1d", top_k=50)
W = qls(sig, UH)
P = W * R.shift(-1)                          # price pnl per coin-day (attributed to decision day)
age = C.notna().cumsum()
tot = P.sum().sum()
rows = []
for lab_, lo, hi in (("<180d", 0, 180), ("180-365d", 180, 365), (">365d", 365, 10 ** 6)):
    m = (age >= lo) & (age < hi)
    rows.append((lab_, P.where(m & (W > 0)).sum().sum() / tot, P.where(m & (W < 0)).sum().sum() / tot,
                 (m & (W != 0)).sum().sum() / (W != 0).sum().sum()))
r5 = C / C.shift(5) - 1
cr = r5 <= -0.25
crow = (P.where(cr & (W > 0)).sum().sum() / tot, P.where(cr & (W < 0)).sum().sum() / tot)
# young coins: PnL per coin-day in leg
young = age < 180
# 2: exclude crash coin-days
res_nc = bt(qls(sig, UH & ~cr))["metrics"]
res_nc20 = bt(qls(sig, UH & ~(r5 <= -0.20)))["metrics"]
# also exclude young (<180d) and crash
res_both = bt(qls(sig, lab.hype_universe("1d", top_k=50, min_hist_days=180) & ~cr))["metrics"]
# 3: non-HYPE ranked among themselves
hs = set(lab.hype_symbols())
qv = lab.load("qv", "1d")
hist = C.notna().cumsum()
nonh = pd.Series(~C.columns.isin(list(hs)), index=C.columns)
elig = C.notna() & (hist >= 60) & nonh
dqv = qv.rolling(30, min_periods=15).median()
rk = dqv.where(elig).rank(axis=1, ascending=False)
UN = elig & (rk <= 30)
resN = bt(qls(sig, UN)); mN = resN["metrics"]; sN = subs(resN["df"]["r"])
UN50 = elig & (rk <= 50)
mN50 = bt(qls(sig, UN50))["metrics"]
# HYPE ranked top-30 for like-for-like breadth
mH30 = bt(qls(sig, lab.hype_universe("1d", top_k=30)))["metrics"]
sol = C.loc["2022-11-04":"2022-11-12", "SOLUSDT"]

L = ["### Step 3b - survivorship follow-ups",
     "Price-PnL share of the base book by coin age (share of total price PnL; share of held coin-days):", "",
     "| age bucket | long-leg PnL share | short-leg PnL share | share of coin-days |", "|---|---|---|---|"]
for a, l_, s_, n_ in rows:
    L.append(f"| {a} | {l_:+.0%} | {s_:+.0%} | {n_:.0%} |")
L += ["",
      f"- Crash coin-days (trailing 5d <= -25%): long-leg PnL share {crow[0]:+.0%}, short-leg {crow[1]:+.0%} of total price PnL.",
      f"- Universe minus crash coin-days (5d<=-25%): Sharpe {res_nc['sharpe']:.2f}; (5d<=-20%): {res_nc20['sharpe']:.2f}; "
      f"min_hist180 AND no crash coin-days: {res_both['sharpe']:.2f}",
      f"- Non-HYPE survivors ranked among themselves, top-30: Sharpe {mN['sharpe']:.2f} (sub {', '.join(f'{k} {v:.2f}' for k, v in sN.items())}), "
      f"npos {mN['avg_n_pos']:.1f}; top-50: {mN50['sharpe']:.2f}. Like-for-like HYPE top-30: {mH30['sharpe']:.2f}.",
      "- SOL closes 2022-11-04..12: " + ", ".join(f"{i.date()} {v:.2f}" for i, v in sol.items())]
append("\n".join(L))
print("\n".join(L))
