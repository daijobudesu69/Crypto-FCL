"""r2_seasonality shared helpers. Data ONLY through lab.py (truncated at 2025-01-01).

Signal inputs allowed in round 2: open/high/low/close/qv/trades. This folder only uses close
(returns) and the point-in-time HYPE liquidity universe (built by lab from close+qv).
Funding is used ONLY as a cost inside lab.backtest.
"""
import os, sys, json
import numpy as np, pandas as pd

sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNLOG = os.path.join(HERE, "runlog.jsonl")
COST = lab.HYPE_COST
SUBP = [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"),
        ("2023-24", "2023-01-01", "2025-01-01")]
EST = ("2020-01-01", "2023-01-01")
VAL = ("2023-01-01", "2025-01-01")
TFD = {"1d": pd.Timedelta("1D"), "4h": pd.Timedelta("4h")}


def ts(s):
    return pd.Timestamp(s, tz="UTC")


def sharpe(d):
    d = pd.Series(d).dropna()
    return float(d.mean() / d.std() * np.sqrt(365)) if len(d) > 5 and d.std() > 0 else np.nan


def daily(res):
    return lab.to_daily(res["df"]["r"], res["tf"])


def sub_sharpes(d):
    out = {}
    for k, a, b in SUBP:
        x = d[(d.index >= ts(a)) & (d.index < ts(b))]
        out[k] = sharpe(x) if len(x) > 30 else np.nan
    return out


def fmt_sub(s):
    return ", ".join(f"{k}: {v:.2f}" for k, v in s.items())


def yearly_str(d):
    y = (1 + d).groupby(d.index.year).prod() - 1
    return ", ".join(f"{k}: {v:+.1%}" for k, v in y.items())


def hac_ols(y, X, lags=10):
    """OLS with Newey-West t-stats; X without constant. Returns (beta, t, n)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    y, X = y[ok], X[ok]
    n = len(y)
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
    return b, b / np.sqrt(np.diag(V)), n


_ctrl = None


def controls():
    """lab.controls() cached to disk (IS-only data)."""
    global _ctrl
    if _ctrl is None:
        p = os.path.join(HERE, "_controls_IS.parquet")
        if os.path.exists(p):
            _ctrl = pd.read_parquet(p)
        else:
            _ctrl = lab.controls()
            _ctrl.to_parquet(p)
    return _ctrl


# ------------------------------------------------------------------ universes / weights
def hype_U(top_k=50):
    return lab.hype_universe("1d", top_k=top_k)


def daily_to_4h(Ud, idx4h):
    """Daily row d is known at close of d (= d+1 00:00). 4h row T is decided at close T+4h,
    so it may use daily row floor(T+4h) - 1D."""
    key = (idx4h + TFD["4h"]).floor("1D") - pd.Timedelta("1D")
    out = Ud.reindex(key)
    out.index = idx4h
    return out.fillna(False).astype(bool)


def topn_ls(score, U, n=8, gross=1.0):
    """Long top-n / short bottom-n by score among valid names (n capped at floor(valid/2)).
    Equal weight, each side gross/2."""
    s = score.where(U)
    valid = s.notna().sum(axis=1)
    k = np.minimum(n, (valid // 2)).astype(float)
    rk_hi = s.rank(axis=1, ascending=False, method="first")
    rk_lo = s.rank(axis=1, ascending=True, method="first")
    lo = rk_hi.le(k, axis=0) & s.notna()
    sh = rk_lo.le(k, axis=0) & s.notna()
    kk = k.replace(0, np.nan)
    W = lo.astype(float).div(kk, axis=0) - sh.astype(float).div(kk, axis=0)
    return W.fillna(0.0) * gross / 2


def quint_ls(score, U, q=0.2, gross=1.0):
    return lab.quantile_ls_weights(score, U, q=q, gross=gross)


def rank_ls(score, U, gross=1.0):
    """Rank-proportional $-neutral weights over all valid names."""
    r = lab.xs_rank(score, U)
    return lab.xs_demean_weights(r, r.notna(), gross=gross)


# ------------------------------------------------------------------ evaluation
def min_equity(W, min_order=10.0):
    a = W.abs().where(W.abs() > 1e-9)
    mn = a.min(axis=1).dropna()
    if len(mn) == 0:
        return np.nan
    return float((min_order / mn).median())


def evaluate(W, tf, name, family, n_trials, full=True, extra=None, save_daily=False):
    """Standard candidate evaluation at HYPE cost. Returns a flat dict and appends to runlog."""
    res = lab.backtest(W, tf, cost=COST)
    m = res["metrics"]
    d = daily(res)
    out = {"name": name, "family": family, "tf": tf, "sharpe": m["sharpe"], "cagr": m["cagr"],
           "maxdd": m["maxdd"], "t_nw": m["t_nw"], "vol": m["vol"],
           "turnover_per_year": m["turnover_per_year"], "cost_drag_per_year": m["cost_drag_per_year"],
           "funding_per_year": m["funding_per_year"], "avg_gross": m["avg_gross"],
           "avg_n_pos": m["avg_n_pos"], "skew": m["skew"], "kurt": m["kurt"], "days": m["days"]}
    g = lab.to_daily(res["df"]["gross_r"] + res["df"]["fund_r"], tf)
    out["sharpe_gross_nocost"] = sharpe(g)
    ss = sub_sharpes(d)
    out.update({f"sh_{k}": v for k, v in ss.items()})
    out["yearly"] = yearly_str(d)
    if full:
        r2 = lab.backtest(W, tf, cost=2 * COST)
        out["sharpe_cost2x"] = r2["metrics"]["sharpe"]
        r3 = lab.backtest(W, tf, cost=COST, extra_lag=1)
        out["sharpe_lag1"] = r3["metrics"]["sharpe"]
        rl = lab.backtest(W.clip(lower=0), tf, cost=COST)
        rs = lab.backtest(W.clip(upper=0), tf, cost=COST)
        out["long_leg_sharpe"] = rl["metrics"].get("sharpe", np.nan)
        out["short_leg_sharpe"] = rs["metrics"].get("sharpe", np.nan)
        c = controls()
        j = pd.concat([d.rename("s"), c], axis=1, join="inner").dropna()
        out["corr_btc"] = float(j["s"].corr(j["btc"]))
        out["corr_xsmom"] = float(j["s"].corr(j["xsmom14"]))
        b, t, n = hac_ols(j["s"].values, j[["xsmom14", "btc"]].values)
        out["alpha_ann_vs_mom_btc"] = float(b[0] * 365)
        out["alpha_t_vs_mom_btc"] = float(t[0])
        out["beta_xsmom"] = float(b[1]); out["beta_btc"] = float(b[2])
        b1, t1, _ = hac_ols(j["s"].values, j[["xsmom14"]].values)
        out["alpha_t_vs_mom"] = float(t1[0])
        out["min_equity_usd"] = min_equity(W.reindex(res["df"].index - TFD[tf]).dropna(how="all"))
        out["dsr"] = lab.deflated_sharpe(m["sharpe"], n_trials, len(d), m["skew"], m["kurt"] + 3)
    if extra:
        out.update(extra)
    row = {k: (round(v, 4) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else v))
           for k, v in out.items()}
    with open(RUNLOG, "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    if save_daily:
        d.to_frame("r").to_parquet(os.path.join(HERE, f"daily_{name}.parquet"))
    return out


def short_line(o):
    s = (f"{o['name']:38s} Sh {o['sharpe']:+.2f} (gross {o['sharpe_gross_nocost']:+.2f})  "
         f"CAGR {o['cagr']:+.1%}  DD {o['maxdd']:.0%}  turn/yr {o['turnover_per_year']:.0f}  "
         f"cost/yr {o['cost_drag_per_year']:.1%}  npos {o['avg_n_pos']:.1f} | "
         f"20-21 {o['sh_2020-21']:+.2f} 22 {o['sh_2022']:+.2f} 23-24 {o['sh_2023-24']:+.2f}")
    if "sharpe_cost2x" in o:
        s += (f" | 2x {o['sharpe_cost2x']:+.2f} lag1 {o['sharpe_lag1']:+.2f} L {o['long_leg_sharpe']:+.2f} "
              f"S {o['short_leg_sharpe']:+.2f} cBTC {o['corr_btc']:+.2f} cMOM {o['corr_xsmom']:+.2f} "
              f"a_t {o['alpha_t_vs_mom_btc']:+.2f}")
    return s
