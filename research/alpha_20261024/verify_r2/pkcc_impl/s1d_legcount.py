"""Step 1d: explain the 0.14 gap. Hypothesis: leg-count convention. pandas pct-rank: 'top q' via r>=1-q takes
floor(qN)+1 names, 'bottom q' via r<=q takes floor(qN). Hunter longs top-q of -ratio (=> floor(qN)+1 LOW-ratio
longs, floor(qN) HIGH-ratio shorts); my reading took the reverse. PRE-REGISTERED 4 conventions + hunter-exact:
 A own (L=floor, S=floor+1)  B hunter (L=floor+1, S=floor)  C both floor  D both floor+1 ; E = hunter W exact (inf kept)
Plus leg excess vs EW universe for A and B."""
import numpy as np, pandas as pd
from common_pk import *
H, L, C = panels()
U = lab.hype_universe("1d", top_k=50)
S = ratio_signal(H, L, C, 30, 24)
M = U & S.notna()
s = S.where(M)
N = M.sum(axis=1)
ra = s.rank(axis=1, method="first")                      # 1 = lowest ratio
rd = s.rank(axis=1, method="first", ascending=False)     # 1 = highest ratio
fq = np.floor(0.2 * N + 1e-9)
def build(nl, ns):
    lo = ra.le(nl, axis=0) & M
    sh = rd.le(ns, axis=0) & M
    ok = (N >= 5)
    lo = lo & ok.values[:, None]; sh = sh & ok.values[:, None]
    W = lo.astype(float).div(lo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5 \
        - sh.astype(float).div(sh.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5
    return W
cfg = {"A own L=f,S=f+1": (fq, fq + 1), "B hunter L=f+1,S=f": (fq + 1, fq), "C both f": (fq, fq), "D both f+1": (fq + 1, fq + 1)}
Ws = {k: build(*v) for k, v in cfg.items()}
# exact hunter W (copied formula, incl. h/l clip, separate means, inf kept)
pc = C.shift(1)
lr_h = np.log((H / L).clip(lower=1.0 + 1e-5))
Sh = -np.sqrt((lr_h**2/(4*np.log(2))).rolling(30, min_periods=24).mean() / (np.log(C/pc)**2).rolling(30, min_periods=24).mean())
r = Sh.where(U).rank(axis=1, pct=True)
lo = (r >= 0.8).astype(float); sh = (r <= 0.2).astype(float)
nl = lo.sum(axis=1).replace(0, np.nan); ns = sh.sum(axis=1).replace(0, np.nan)
WE = (lo.div(nl, axis=0) - sh.div(ns, axis=0)).where(nl.notna() & ns.notna(), 0).fillna(0) * 0.5
Ws["E hunter exact"] = WE
ew = M.astype(float).div(M.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
rew = own_bt(ew)["r"]
lines = []
for k, W in Ws.items():
    df = own_bt(W); rr = df["r"]
    y = yearly(rr)
    sp = subp(rr)
    l2 = own_bt(W, extra_lag=1)["r"]; c2 = own_bt(W, cost=2*lab.HYPE_COST)["r"]
    WL = W.clip(lower=0); WS = -W.clip(upper=0)
    rl = own_bt(WL.div(WL.sum(axis=1).replace(0, np.nan), axis=0).fillna(0))["r"]
    rs = own_bt(WS.div(WS.sum(axis=1).replace(0, np.nan), axis=0).fillna(0))["r"]
    j = pd.concat([rl.rename("l"), rs.rename("s"), rew.rename("e")], axis=1).dropna()
    n_l = (W > 0).sum(axis=1); n_s = (W < 0).sum(axis=1); a = n_l + n_s > 0
    line = (f"{k:22s} SR {sharpe(rr):.2f} MDD {maxdd(rr):.1%} sub {sp['2020-21']:.2f}/{sp['2022']:.2f}/{sp['2023-24']:.2f} "
            f"2x {sharpe(c2):.2f} lag1 {sharpe(l2):.2f} | yrs " + " ".join(f"{yy}:{v:+.0%}" for yy, v in y['ret'].items()) +
            f" | nL {n_l[a].mean():.1f} nS {n_s[a].mean():.1f} | xsSR long {sharpe(j['l']-j['e']):.2f} short {sharpe(j['e']-j['s']):.2f}")
    print(line); lines.append(line)
d = (Ws["E hunter exact"] - Ws["B hunter L=f+1,S=f"]).abs().sum(axis=1)
line = f"E vs B weight diff: days with any diff {(d>1e-9).sum()}, mean L1 diff on those {d[d>1e-9].mean():.3f}"
print(line); lines.append(line)
open("s1d_out.txt", "w").write("\n".join(lines))
