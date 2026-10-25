"""Small helpers shared by famB_post*.py (identical to the definitions in famB_post.py)."""
import numpy as np, pandas as pd
from famB_regime import C, trail_pct, state


def disp(U, L=7, min_hist=None):
    m = U
    if min_hist is not None:
        m = U & (C.notna().cumsum() >= min_hist)
    return np.log(C / C.shift(L)).where(m).std(axis=1)


def st_from(sig, win=365, kind="MED", sign=1):
    p = trail_pct(sig, win=win, minp=min(120, win // 2))
    if sign < 0:
        p = 1 - p
    return state(p, kind)


def align(s, idx):
    s2 = s.copy(); s2.index = s2.index + pd.Timedelta("1D")
    return s2.reindex(idx)
