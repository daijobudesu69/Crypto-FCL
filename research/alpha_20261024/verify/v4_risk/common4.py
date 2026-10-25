"""v4_risk helpers. Research period 2020-01-01..2024-12-31 only, all data via lab.py (no ALPHA_FORWARD)."""
import sys, os
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
assert os.environ.get("ALPHA_FORWARD") != "1", "never run verification with ALPHA_FORWARD"


def sh(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) > 20 and x.std() > 0 else np.nan


def candidate_weights(U=None, F=None, q=0.33):
    """h1 candidate A4_L14_top50_tercile_rb1: short top third by F14, long bottom third, 0.5 gross per side."""
    F = lab.load("fund", "1d") if F is None else F
    U = lab.universe("1d", top_k=50) if U is None else U
    s = -F.rolling(14, min_periods=14).sum()
    return lab.quantile_ls_weights(s, U, q=q), s


def hourly_cube(close_1d):
    """(n_days, 24, n_coins) float32 array of 1h closes for each 1d bar (rows aligned with close_1d.index).
    Hour k of day D = close of the 1h bar opening at D+k h, so hour 23 = the daily close."""
    c1 = lab.load("close", "1h")
    c1 = c1.reindex(columns=close_1d.columns)
    ix = pd.date_range(close_1d.index[0], close_1d.index[-1] + pd.Timedelta("23h"), freq="1h")
    c1 = c1.reindex(ix)
    a = c1.values.astype("float32").reshape(len(close_1d.index), 24, -1)
    return a


def sim(W, tf="1d", cost=lab.COST, stop=None, stop_slip=0.005, close=None, high=None, fund=None, cube=None):
    """Daily simulator equal to lab.backtest (drift-adjusted turnover cost + funding) with an optional
    intraday stop on SHORT positions: stop = X means a short is closed when the bar high reaches
    entry*(1+X) (entry = close of the bar at which the current short episode was opened), fill at
    entry*(1+X)*(1+stop_slip). After a stop the name is banned from the short leg until the signal
    itself drops it (episode end). The freed weight stays in cash (no redistribution).
    Returns DataFrame indexed like lab.backtest's df (realisation bar)."""
    C = lab.load("close", tf) if close is None else close
    H = lab.load("high", tf) if high is None else high
    Fd = lab.load("fund", tf) if fund is None else fund
    W = W.reindex(index=C.index, columns=C.columns).fillna(0.0)
    c = C.values.astype(float); h = H.reindex_like(C).values.astype(float); f = Fd.reindex_like(C).fillna(0).values.astype(float)
    w_t = W.values.copy()
    n, m = w_t.shape
    rn = np.full((n, m), np.nan); rn[:-1] = c[1:] / c[:-1] - 1
    hn = np.full((n, m), np.nan); hn[:-1] = h[1:]
    fn = np.zeros((n, m)); fn[:-1] = f[1:]
    entry = np.full(m, np.nan)      # entry price of current short episode
    banned = np.zeros(m, bool)
    prev_sig_short = np.zeros(m, bool)
    prev_w = np.zeros(m); prev_r = np.zeros(m); prev_port = 0.0
    out = np.zeros((n, 7))
    nstops = 0
    for t in range(n):
        sig = w_t[t].copy()
        sig_short = sig < 0
        # episode bookkeeping
        new_ep = sig_short & ~prev_sig_short
        entry[new_ep] = c[t, new_ep]
        ended = ~sig_short
        banned[ended] = False
        entry[ended] = np.nan
        w = sig.copy()
        w[banned] = 0.0
        ok = ~np.isnan(rn[t])
        w[~ok] = 0.0
        r = np.where(ok, rn[t], 0.0)
        stopped_now = np.zeros(m, bool)
        if stop is not None and t + 1 < n:
            lvl = entry * (1 + stop)
            if cube is None:
                hit = (w < 0) & ok & ~np.isnan(hn[t]) & (hn[t] >= lvl)
                px = np.maximum(lvl, c[t]) * (1 + stop_slip)
            else:
                # trigger on 1h closes of the holding bar (bar t+1); fill at the first 1h close >= level
                hc = cube[t + 1]                         # (24, m)
                with np.errstate(invalid="ignore"):
                    cross = hc >= lvl[None, :]
                anyc = cross.any(axis=0)
                hit = (w < 0) & ok & anyc
                first = np.argmax(cross, axis=0)
                px = hc[first, np.arange(m)].astype(float) * (1 + stop_slip)
            if hit.any():
                r = r.copy(); r[hit] = px[hit] / c[t, hit] - 1
                stopped_now = hit
                nstops += int(hit.sum())
        gross = (w * r).sum()
        fund_r = -(w * np.where(stopped_now, 0.5, 1.0) * fn[t]).sum()   # stopped names pay ~half a day of funding
        drift = prev_w * (1 + prev_r) / (1 + prev_port)
        turn = np.abs(w - drift).sum()
        # stopped positions: exit trade happens intraday (another side of cost) and are 0 at next bar
        turn_stop = np.abs(w[stopped_now] * (1 + r[stopped_now]) / (1 + gross)).sum() if stopped_now.any() else 0.0
        cost_r = (turn + turn_stop) * cost
        net = gross + fund_r - cost_r
        out[t] = [net, gross, fund_r, cost_r, turn + turn_stop, np.abs(w).sum(), (w != 0).sum()]
        # after a stop: position is flat for the rest of the bar and banned
        banned |= stopped_now
        w_after = w.copy(); w_after[stopped_now] = 0.0
        prev_w = w_after; prev_r = r; prev_port = gross
        prev_sig_short = sig_short
    df = pd.DataFrame(out, columns=["r", "gross_r", "fund_r", "cost_r", "turnover", "gross_exposure", "n_pos"], index=C.index)
    df.index = df.index + pd.Timedelta(lab.TF_DELTA[tf])
    df = df.iloc[:-1]
    nz = np.flatnonzero(df["gross_exposure"].values > 0)
    df = df.iloc[nz[0]:] if len(nz) else df
    df.attrs["nstops"] = nstops
    return df


def summarize(r):
    m = lab.metrics(r, "1d")
    y = lab.yearly(r, "1d")
    return {"sharpe": m["sharpe"], "cagr": m["cagr"], "maxdd": m["maxdd"], "vol": m["vol"], "skew": m["skew"],
            "kurt": m["kurt"], "worst_month": m["worst_month"], "days": m["days"],
            "yearly_ret": {int(k): round(float(v), 4) for k, v in y["ret"].items()},
            "yearly_sh": {int(k): round(float(v), 2) for k, v in y["sharpe"].items()}}
