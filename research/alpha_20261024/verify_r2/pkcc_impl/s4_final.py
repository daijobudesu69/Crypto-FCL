"""Step 4: stress of the ONE selected $200 config (no new selection): 2x cost, fresh $200 starts 2021/2022,
lag2, and deflated Sharpe at the desk trial count. Usage: python s4_final.py D?"""
import sys, json
import numpy as np, pandas as pd
import s3_design as s3   # re-uses panels + dyn(); s3 grid is NOT re-run (guarded below)

cid = sys.argv[1]
G, kmax, band, reb = s3.CFG[cid]
lines = []
def P(s):
    print(s, flush=True); lines.append(s)
for nm, kw in (("2x cost full", {"cost2": True}), ("fresh 2021-01-01", {"start": "2021-01-01"}),
               ("fresh 2022-01-01", {"start": "2022-01-01"}), ("lag2 full", {"lag": 2})):
    old = s3.COST
    if kw.pop("cost2", False):
        s3.COST = 2 * old
    e, Wd, info = s3.dyn(G, kmax, band, reb, **kw)
    s3.COST = old
    st = s3.stats(e)
    P(f"{cid} {nm}: SR {st['sr']:.2f} $200->{st['final']:.0f} MDD {st['mdd']:.0%} minEq {st['min_eq']:.0f} "
      f"yrs {' '.join(f'{k}:{v:+.0%}' for k, v in st['yrs'].items())}")
# DSR on the full-run daily equity returns
e, Wd, info = s3.dyn(G, kmax, band, reb)
r = e.pct_change().dropna()
from common_pk import lab
sk = float(r.skew()); ku = float(r.kurt()) + 3
for n in (855, 855 + 37):
    P(f"{cid} DSR n={n}: {lab.deflated_sharpe(s3.sharpe(r), n, len(r), sk, ku):.3f} (SR {s3.sharpe(r):.2f}, skew {sk:.2f}, kurt {ku:.1f})")
# same for the hunter's headline quintile book (frictionless lab numbers)
P(f"hunter quintile book SR 1.34 skew 0.42 kurt 11.3 T 1758: DSR n=855 {lab.deflated_sharpe(1.34, 855, 1758, 0.42, 11.32):.3f}, "
  f"n=892 {lab.deflated_sharpe(1.34, 892, 1758, 0.42, 11.32):.3f}; own-convention SR 1.20: {lab.deflated_sharpe(1.20, 892, 1758, 0.42, 11.32):.3f}")
open(f"s4_out_{cid}.txt", "w").write("\n".join(lines))
