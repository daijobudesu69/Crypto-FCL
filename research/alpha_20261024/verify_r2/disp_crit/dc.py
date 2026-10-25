"""disp_crit verifier helpers -- independent re-implementation of DISP7 (hunter r2_age_dispersion).

Data ONLY through lab.py (truncated at 2025-01-01). ALPHA_FORWARD is never set.
Signal inputs: close (+ qv via lab universe). Funding charged as cost by lab.backtest.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab  # noqa: E402

assert os.environ.get("ALPHA_FORWARD") != "1", "forward data must not be enabled"
HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "REPORT.md")
BTC = "BTCUSDT"
COST = lab.HYPE_COST
C = lab.load("close", "1d")
R = C.pct_change(fill_method=None)
assert C.index.max() < lab.IS_END
FIRST = C.apply(lambda s: s.first_valid_index())
HIST = C.notna().cumsum()


def U50(top_k=50):
    return lab.hype_universe("1d", top_k=top_k)


def disp(U, L=7, kind="std"):
    x = np.log(C / C.shift(L)).where(U)
    if kind == "std":
        return x.std(axis=1)
    if kind == "iqr":
        return (x.quantile(0.75, axis=1) - x.quantile(0.25, axis=1)) / 1.349
    if kind == "mad":
        med = x.median(axis=1)
        return x.sub(med, axis=0).abs().median(axis=1) * 1.4826
    raise ValueError(kind)


def trail_pct(s, win=365, minp=None):
    """percentile of s[t] among the finite values of s[t-win+1 .. t] (trailing only, includes t)."""
    minp = min(120, win // 2) if minp is None else minp
    v = s.values.astype(float)
    out = np.full(len(v), np.nan)
    for t in range(len(v)):
        if not np.isfinite(v[t]):
            continue
        w = v[max(0, t - win + 1):t + 1]
        w = w[np.isfinite(w)]
        if len(w) >= minp:
            out[t] = (w <= v[t]).mean()
    return pd.Series(out, index=s.index)


def terc_state(p, hi=2 / 3, lo=1 / 3):
    cur = np.nan
    out = np.full(len(p), np.nan)
    for i, v in enumerate(p.values):
        if np.isfinite(v):
            if v > hi:
                cur = 1.0
            elif v < lo:
                cur = -1.0
        out[i] = cur
    return pd.Series(out, index=p.index)


def spread_w(st, U, alt_mask=None):
    alts = U.copy()
    if alt_mask is not None:
        alts &= alt_mask
    alts[BTC] = False
    Wa = alts.astype(float).div(alts.sum(1).replace(0, np.nan), axis=0).fillna(0.0) * 0.5
    s = st.reindex(U.index).fillna(0.0)
    W = Wa.mul(s, axis=0)
    W[BTC] = -0.5 * s
    return W


def disp7_state(U=None, L=7, win=365, hi=2 / 3, lo=1 / 3, kind="std"):
    U = U50() if U is None else U
    return terc_state(trail_pct(disp(U, L, kind), win=win), hi, lo)


def bt(W, cost=COST, **kw):
    return lab.backtest(W, "1d", cost=cost, **kw)


def sharpe(r):
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std() * np.sqrt(365)) if len(r) > 20 and r.std() > 0 else np.nan


def between(r, a=None, b=None):
    m = pd.Series(True, index=r.index)
    if a:
        m &= r.index >= pd.Timestamp(a, tz="UTC")
    if b:
        m &= r.index < pd.Timestamp(b, tz="UTC")
    return r[m]


def subs(r):
    return {"20-21": sharpe(between(r, "2020-01-01", "2022-01-01")), "22": sharpe(between(r, "2022-01-01", "2023-01-01")),
            "23-24": sharpe(between(r, "2023-01-01", "2025-01-01"))}


def yearly_str(r):
    y = lab.yearly(r, "1d")
    return "; ".join(f"{i}: {v:+.1%} (SR {s:.2f})" for i, v, s in zip(y.index, y["ret"], y["sharpe"]))


def align_state(st, idx):
    """state decided at close t applies to the return dated t+1 (backtest index)."""
    s2 = st.copy()
    s2.index = s2.index + pd.Timedelta("1D")
    return s2.reindex(idx)


def append(md):
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(md.rstrip() + "\n\n")
    print(md)
