"""Helpers for hunter h2_positioning. Research data only via lab.py (truncated at 2025-01-01)."""
import sys, os, json
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "runlog.csv")

def tsz(X, L, minp=None):
    """Trailing time-series z-score per coin (no lookahead)."""
    minp = minp or max(10, L // 2)
    m = X.rolling(L, min_periods=minp).mean()
    s = X.rolling(L, min_periods=minp).std()
    return (X - m) / s.where(s > 0)

def lchg(X, k):
    """log change over k bars, trailing."""
    lx = np.log(X.where(X > 0))
    return lx - lx.shift(k)

def qls(score, U, q=0.2, min_names=20, long_only=False, short_only=False):
    """Quintile L/S, but flat on days with < min_names valid scores inside U."""
    s = score.where(U)
    ok = s.notna().sum(axis=1) >= min_names
    s = s.where(np.broadcast_to(ok.values[:, None], s.shape))
    return lab.quantile_ls_weights(s, s.notna(), q=q, long_only=long_only, short_only=short_only)

_CTRL = None
def controls():
    global _CTRL
    if _CTRL is None:
        _CTRL = lab.controls()
    return _CTRL

def sub_sharpe(r, a, b):
    x = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))].dropna()
    if len(x) < 60 or x.std() == 0:
        return np.nan
    return float(x.mean() / x.std() * np.sqrt(365))

def min_equity(W):
    a = W.abs().where(W.abs() > 0)
    mn = a.min(axis=1).dropna()
    if len(mn) == 0:
        return np.nan
    return float(np.median(10.0 / mn))

def evaluate(name, family, W, tf="1d", start=None, full=True, n_note=""):
    """Run base backtest + stress tests; append a row to runlog.csv; return (row, res)."""
    res = lab.backtest(W, tf, start=start)
    m = res["metrics"]
    r = lab.to_daily(res["df"]["r"], tf)
    row = {"name": name, "family": family, "start": m.get("start"), "end": m.get("end"),
           "sharpe": m.get("sharpe"), "cagr": m.get("cagr"), "maxdd": m.get("maxdd"), "t_nw": m.get("t_nw"),
           "turn_yr": m.get("turnover_per_year"), "cost_yr": m.get("cost_drag_per_year"),
           "fund_yr": m.get("funding_per_year"), "npos": m.get("avg_n_pos"),
           "skew": m.get("skew"), "kurt": m.get("kurt")}
    g = lab.to_daily(res["df"]["gross_r"], tf)
    row["gross_price_yr"] = float(res["df"]["gross_r"].mean() * lab.BARS_PER_YEAR[tf])
    for a, b, lab_ in [("2020-01-01", "2022-01-01", "s2021"), ("2022-01-01", "2023-01-01", "s2022"),
                       ("2023-01-01", "2025-01-01", "s2324"), ("2023-01-01", "2024-01-01", "s2023"),
                       ("2024-01-01", "2025-01-01", "s2024")]:
        row[lab_] = sub_sharpe(r, a, b)
    if full:
        row["sh_cost2x"] = lab.backtest(W, tf, cost=0.0014, start=start)["metrics"].get("sharpe")
        row["sh_lag1"] = lab.backtest(W, tf, extra_lag=1, start=start)["metrics"].get("sharpe")
        st = m.get("start")
        rl = lab.backtest(W.clip(lower=0), tf, start=st)
        rs = lab.backtest(W.clip(upper=0), tf, start=st)
        row["sh_long"] = rl["metrics"].get("sharpe"); row["sh_short"] = rs["metrics"].get("sharpe")
        row["ann_long"] = float(rl["df"]["r"].mean() * lab.BARS_PER_YEAR[tf]); row["ann_short"] = float(rs["df"]["r"].mean() * lab.BARS_PER_YEAR[tf])
        c = controls().reindex(r.index)
        row["corr_btc"] = float(r.corr(c["btc"])); row["corr_xsmom"] = float(r.corr(c["xsmom14"]))
        row["min_eq"] = min_equity(W.loc[W.index >= pd.Timestamp(st, tz="UTC")])
    row["note"] = n_note
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    return row, res

def fmt(row):
    keys = ["sharpe", "s2021", "s2022", "s2023", "s2024", "sh_cost2x", "sh_lag1", "sh_long", "sh_short",
            "corr_btc", "corr_xsmom", "turn_yr", "fund_yr", "gross_price_yr", "npos", "maxdd"]
    return f"{row['name']:38s} [{row['start']}] " + " ".join(
        f"{k}={row[k]:.2f}" if isinstance(row.get(k), (float, int)) and row.get(k) == row.get(k) else f"{k}=nan" for k in keys if k in row)

def xs_ic(score, fwd, U, min_names=20):
    """Daily cross-sectional Spearman IC between score[t] and fwd[t] (fwd must already be aligned
    as the return realised AFTER t). Returns (mean IC, NW t-stat, n days)."""
    s = score.where(U & fwd.notna()); f = fwd.where(s.notna())
    ok = s.notna().sum(axis=1) >= min_names
    s = s[ok].rank(axis=1); f = f[ok].rank(axis=1)
    s = s.sub(s.mean(axis=1), axis=0); f = f.sub(f.mean(axis=1), axis=0)
    ic = (s * f).sum(axis=1) / np.sqrt((s ** 2).sum(axis=1) * (f ** 2).sum(axis=1))
    ic = ic.dropna()
    return float(ic.mean()), float(lab.newey_west_t(ic.values, lags=10)), len(ic)

def xs_corr(a, b, U, min_names=20):
    """Average daily cross-sectional Spearman correlation between two signals."""
    m, _, _ = xs_ic(a, b, U, min_names)
    return m

def xs_resid(score, ctrl, U):
    """Cross-sectional residual of rank(score) on rank(ctrl) each day (inside U). No lookahead."""
    s = score.where(U & ctrl.notna()).rank(axis=1, pct=True)
    m = ctrl.where(s.notna()).rank(axis=1, pct=True)
    s = s.sub(s.mean(axis=1), axis=0); m = m.sub(m.mean(axis=1), axis=0)
    b = (s * m).sum(axis=1) / (m * m).sum(axis=1)
    return s - m.mul(b, axis=0)
