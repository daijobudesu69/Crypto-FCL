"""Diagnostic only: yearly stability of the legs of BR_SKEWrev_4h30_DN_R1 (excess vs EW universe)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import lab, ew_return, BTC
C = lab.load("close", "1d"); R = C.pct_change(fill_method=None)
U = lab.universe("1d", min_hist_days=60, top_k=50); U[BTC] = False
sig = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), "famB_rev_signals.pkl"))
from common import rolling_beta, shrink_beta
b60, _, _ = rolling_beta(R, R[BTC], 60); bshr = shrink_beta(b60.where(U))
for s in ["SKEWrev_4h30", "SKEWrev_d30"]:
    x = sig[s]
    W = lab.quantile_ls_weights(x, U & x.notna() & bshr.notna(), q=0.2)
    ew = ew_return(R, U); ew.index = ew.index + pd.Timedelta("1D")
    out = {}
    for leg, Wl in [("long", W.clip(lower=0)), ("short", W.clip(upper=0))]:
        r = lab.backtest(Wl, "1d")["df"]["r"]
        gl = pd.Series(Wl.abs().sum(1).values, index=ew.index).reindex(r.index)
        ex = r - np.sign(Wl.values.sum()) * gl * ew.reindex(r.index)
        out[leg] = ex.groupby(ex.index.year).apply(lambda z: z.mean() / z.std() * np.sqrt(365)).round(2).to_dict()
        out[leg + "_ann"] = ex.groupby(ex.index.year).mean().mul(365).round(3).to_dict()
    print(s); [print("  ", k, v) for k, v in out.items()]
