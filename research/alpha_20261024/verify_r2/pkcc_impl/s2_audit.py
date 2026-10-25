"""Step 2: lookahead audit.
PRE-REGISTERED checks (no parameter search):
  (a) truncation test at 3 cuts: 2021-05-19, 2022-11-08, 2024-03-14 -> U, signal, W and daily returns before the cut
      must be bit-identical to the full-sample run (own convention A and hunter convention B).
  (b) static audit: rolling windows trailing (no center), universe inputs, first valid date, names per leg by year,
      stale-price cells (H==L or zero cc return) inside the traded book.
  (c) execution delay on 1h closes: execute at 00:00+dh, dh = 1, 2, 4 h (cron lateness), hold 24h. 3 configs x 2 conv.
"""
import numpy as np, pandas as pd
from common_pk import *

lines = []
def P(s):
    print(s, flush=True); lines.append(s)

def build_W(conv="A"):
    H, L, C = panels()
    U = lab.hype_universe("1d", top_k=50)
    S = ratio_signal(H, L, C, 30, 24)
    M = U & S.notna()
    if conv == "A":
        W = ls_weights(S, M, q=0.2)
    else:  # hunter convention: long top-q of -ratio, short bottom-q of -ratio
        r = (-S).where(M).rank(axis=1, pct=True)
        lo = (r >= 0.8).astype(float); sh = (r <= 0.2).astype(float)
        nl = lo.sum(axis=1).replace(0, np.nan); ns = sh.sum(axis=1).replace(0, np.nan)
        W = (lo.div(nl, axis=0) - sh.div(ns, axis=0)).where(nl.notna() & ns.notna(), 0).fillna(0) * 0.5
    return U, S, W

FIELDS = ["high", "low", "close", "qv", "fund"]
full = {f: lab.load(f, "1d").copy() for f in FIELDS}
ref = {}
for conv in ("A", "B"):
    U, S, W = build_W(conv)
    ref[conv] = (U, S, W, own_bt(W)["r"])

P("## (a) truncation test")
for cut in ("2021-05-19", "2022-11-08", "2024-03-14"):
    ct = pd.Timestamp(cut, tz="UTC")
    lab.clear()
    for f in FIELDS:
        lab._mem[(f, "1d")] = full[f][full[f].index < ct]
    for conv in ("A", "B"):
        U, S, W = build_W(conv)
        r = own_bt(W)["r"]
        Uf, Sf, Wf, rf = ref[conv]
        ix = W.index
        dU = int((U != Uf.loc[ix]).sum().sum())
        a, b = S.values, Sf.loc[ix].values
        dS = int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
        dW = float(np.abs(W.values - Wf.loc[ix].values).max())
        rr = rf.loc[r.index]
        dr = float(np.abs(r.values - rr.values).max())
        lastW = float(np.abs(W.iloc[-1].values - Wf.loc[ix[-1]].values).max())
        P(f"cut {cut} conv {conv}: rows {len(ix)}, U diff cells {dU}, signal diff cells {dS}, max|dW| {dW:.1e} "
          f"(last row {ix[-1].date()} max|dW| {lastW:.1e}, n_pos {(W.iloc[-1]!=0).sum()}), "
          f"returns compared {len(r)} (last {r.index[-1].date()}), max|dr| {dr:.1e}")
lab.clear()

P("## (b) static audit")
H, L, C = panels()
U, S, W = ref["A"][0], ref["A"][1], ref["A"][2]
WB = ref["B"][2]
act = (W != 0).any(axis=1)
P(f"first eligible date (U & ratio valid & N>=5): {((U & S.notna()).sum(axis=1) >= 5).idxmax().date()}, first trade {act.idxmax().date()}")
for y in range(2020, 2025):
    m = act & (W.index.year == y)
    nl = (WB[m] > 0).sum(axis=1); ns = (WB[m] < 0).sum(axis=1)
    P(f"{y}: hunter-conv names/leg long {nl.mean():.1f} (min {nl.min()}) short {ns.mean():.1f} (min {ns.min()})")
# stale prices in the book
stale = (H == L)
pc = C.shift(1)
zc = (C == pc)
for nm, WW in (("A", W), ("B", WB)):
    held = WW != 0
    P(f"conv {nm}: book cells with H==L that day {int((held & stale).sum().sum())}, "
      f"book cells whose 30d window has >=10 H==L days {int((held & (stale.astype(float).rolling(30).sum() >= 10)).sum().sum())}, "
      f"ICP held days {int(held['ICPUSDT'].sum()) if 'ICPUSDT' in held else 0}")
# min-periods usage: cells in book with < 30 joint obs
pk = (np.log(H / L)) ** 2; cc = np.log(C / pc) ** 2
nobs = (pk.notna() & cc.notna()).astype(float).rolling(30).sum()
P(f"book cells with <30 joint obs in window: {int(((W != 0) & (nobs < 30)).sum().sum())} of {int((W != 0).sum().sum())}")
# universe membership: listed-on-HYPE-today filter
hs = lab.hype_symbols()
P(f"HYPE symbol list = coins on HYPE in 2026 ({len(hs)}); applied to all of 2020-24 -> future-listing selection (not price lookahead)")

P("## (c) execution delay on 1h closes")
C1 = lab.load("close", "1h").astype("float64")
Cd = C.copy()
for dh in (0, 1, 2, 4):
    # price at time D + 1day + dh  (= close of 1h bar with open ts D + 1day + dh - 1h)
    tgt_ts = Cd.index + pd.Timedelta(days=1) + pd.Timedelta(hours=dh) - pd.Timedelta(hours=1)
    Cx = C1.reindex(tgt_ts)
    Cx.index = Cd.index
    Cx = Cx.reindex(columns=Cd.columns)
    if dh == 0:
        d = (Cx / Cd - 1).abs()
        P(f"consistency 1h close at 23:00 bar vs 1d close: median |diff| {np.nanmedian(d.values):.1e}, p99 {np.nanpercentile(d.values, 99):.1e}")
    for nm, WW in (("A", W), ("B", WB)):
        r = own_bt(WW, C=Cx)["r"]
        sp = subp(r)
        P(f"delay {dh}h conv {nm}: SR {sharpe(r):.2f} sub {sp['2020-21']:.2f}/{sp['2022']:.2f}/{sp['2023-24']:.2f}")
del C1
lab.clear()
open("s2_out.txt", "w").write("\n".join(lines))
