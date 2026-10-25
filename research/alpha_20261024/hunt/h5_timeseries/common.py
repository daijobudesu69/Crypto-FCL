"""h5_timeseries shared helpers (no data access beyond lab.py)."""
import sys
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import numpy as np
import pandas as pd
import lab

EST = ("2020-01-01", "2023-01-01")     # estimation window (inside IS)
VAL = ("2023-01-01", "2025-01-01")     # validation window (inside IS, never used to choose)
SUBP = {"2020-21": ("2020-01-01", "2022-01-01"), "2022": ("2022-01-01", "2023-01-01"),
        "2023-24": ("2023-01-01", "2025-01-01")}
TF_DELTA = {"1d": pd.Timedelta("1D"), "4h": pd.Timedelta("4h"), "1h": pd.Timedelta("1h")}


def ts(s):
    return pd.Timestamp(s, tz="UTC")


def daily_universe(top_k=20, min_hist_days=60):
    return lab.universe("1d", min_hist_days=min_hist_days, top_k=top_k)


def map_daily_to_intraday(Ud, intraday_index, tf):
    """Daily panel row d is known at close of d (= d+1 00:00). Intraday row T is decided at
    close T+tf, so it may use daily row floor(T+tf) - 1D."""
    key = (intraday_index + TF_DELTA[tf]).floor("1D") - pd.Timedelta("1D")
    out = Ud.reindex(key)
    out.index = intraday_index
    return out


def ew_weights(U):
    U = U.astype(float)
    return U.div(U.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def basket_returns(U, tf):
    """Equal-weight return of bar t+1 for members decided at close t, indexed by bar t+1 (open ts).
    Simple descriptive series (rebalanced each bar, no costs)."""
    R = lab.returns(tf)
    M = U.reindex(index=R.index, columns=R.columns).fillna(False).astype(bool).shift(1, fill_value=False)
    M = M & R.notna()
    n = M.sum(axis=1)
    r = R.where(M).sum(axis=1) / n.replace(0, np.nan)
    return r   # value at index t = EW return of bar t (members fixed at close t-1)


def drift_hold(Wt, R, rebal_mask):
    """Hold weights with drift between rebalance bars. Wt = target weights (used only where
    rebal_mask True). R = bar returns panel (known at close of bar t). Path-dependent, past-only."""
    cols = Wt.columns
    R = R.reindex(index=Wt.index, columns=cols).fillna(0.0).values
    T = Wt.fillna(0.0).values
    m = rebal_mask.reindex(Wt.index).fillna(False).values
    out = np.zeros_like(T)
    w = np.zeros(T.shape[1])
    for t in range(len(T)):
        if t > 0:
            gr = w * (1 + R[t])
            tot = 1 + (w * R[t]).sum()
            w = gr / tot if tot > 0 else gr
        if m[t]:
            w = T[t].copy()
        out[t] = w
    return pd.DataFrame(out, index=Wt.index, columns=cols)


def sharpe(d):
    d = d.dropna()
    return float(d.mean() / d.std() * np.sqrt(365)) if d.std() > 0 else np.nan


def daily(res):
    return lab.to_daily(res["df"]["r"], res["tf"])


def sub_sharpes(d):
    out = {}
    for k, (a, b) in SUBP.items():
        x = d[(d.index >= ts(a)) & (d.index < ts(b))]
        out[k] = sharpe(x) if len(x) > 30 else np.nan
    return out


def fmt_sub(s):
    return ", ".join(f"{k}: {v:.2f}" for k, v in s.items())


def yearly_str(d):
    y = (1 + d).groupby(d.index.year).prod() - 1
    return ", ".join(f"{k}: {v:+.1%}" for k, v in y.items())


def hac_t(y, X, lags=None):
    """OLS with Newey-West t-stats. X without constant (added). Returns beta, t."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    if lags is None:
        lags = int(4 * (n / 100) ** (2 / 9))
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    t = b / np.sqrt(np.diag(V))
    return b, t, n


def full_eval(W, tf, name, n_trials, ctrl=None, extra=None):
    """Standard evaluation dict for a candidate."""
    res = lab.backtest(W, tf)
    m = res["metrics"]
    d = daily(res)
    r2 = lab.backtest(W, tf, cost=0.0014)
    rl = lab.backtest(W, tf, extra_lag=1)
    out = {"name": name, "tf": tf, "sharpe": m["sharpe"], "cagr": m["cagr"], "maxdd": m["maxdd"],
           "t_nw": m["t_nw"], "turnover_per_year": m["turnover_per_year"],
           "cost_drag_per_year": m["cost_drag_per_year"], "funding_per_year": m["funding_per_year"],
           "avg_n_pos": m["avg_n_pos"], "avg_gross": m["avg_gross"],
           "sharpe_cost2x": r2["metrics"]["sharpe"], "sharpe_lag1": rl["metrics"]["sharpe"],
           "sub": sub_sharpes(d), "yearly": yearly_str(d), "start": m["start"], "end": m["end"],
           "gross_sharpe": sharpe(lab.to_daily(res["df"]["gross_r"], tf)),
           "dsr": lab.deflated_sharpe(m["sharpe"], n_trials, m["days"], m["skew"], m["kurt"] + 3)}
    if ctrl is not None:
        j = pd.concat([d.rename("s"), ctrl], axis=1).dropna()
        out["corr_btc"] = float(j["s"].corr(j["btc"]))
        out["corr_xsmom"] = float(j["s"].corr(j["xsmom14"]))
    if extra:
        out.update(extra)
    return out, res
