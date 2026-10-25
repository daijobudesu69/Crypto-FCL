"""Shared helpers for hunter r2_age_dispersion (round 2: HYPE candles only).

Data ONLY through lab.py (truncated at 2025-01-01). Signal inputs restricted to
open/high/low/close/qv/trades. Funding is charged as a cost inside lab.backtest.
All rolling statistics are trailing only.
"""
import os, sys, csv, json
import numpy as np, pandas as pd

sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNLOG = os.path.join(HERE, "runlog.csv")
COST = lab.HYPE_COST
BTC = "BTCUSDT"
SUBP = [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"),
        ("2023-24", "2023-01-01", "2025-01-01")]
_ctrl = None


def controls():
    global _ctrl
    if _ctrl is None:
        _ctrl = lab.controls()
    return _ctrl


def sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(365)) if len(r) > 20 and r.std() > 0 else np.nan


def sub_sharpes(r):
    out = {}
    for nm, a, b in SUBP:
        x = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]
        out[nm] = sharpe(x)
    return out


def ols_nw(y, X, lags=10):
    """OLS y = a + X b with Newey-West t-stats. Returns (coef, t)."""
    y = np.asarray(y, float); X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    m = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[m], X[m]
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xe[L:].T @ Xe[:-L] / n
        S += w * (G + G.T)
    V = n * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return b, b / se


def alpha_vs_controls(r):
    """Daily alpha (annualised) and NW t-stat of r on [xsmom14, btc]; plus on xsmom14 alone."""
    c = controls()
    df = pd.concat([r.rename("r"), c], axis=1, join="inner").dropna()
    b, t = ols_nw(df["r"].values, df[["xsmom14", "btc"]].values)
    b1, t1 = ols_nw(df["r"].values, df[["xsmom14"]].values)
    return {"alpha_ann": float(b[0] * 365), "alpha_t_vs_xsmom_btc": float(t[0]),
            "beta_xsmom": float(b[1]), "beta_btc": float(b[2]),
            "alpha_t_vs_xsmom": float(t1[0]),
            "corr_btc": float(df["r"].corr(df["btc"])), "corr_xsmom": float(df["r"].corr(df["xsmom14"]))}


def ew_bench(U):
    """Equal-weight return of universe U, aligned with backtest df index (gross, no cost)."""
    W = U.astype(float)
    W = W.div(W.sum(1).replace(0, np.nan), axis=0).fillna(0.0)
    return lab.backtest(W, "1d", cost=0.0, funding=False)["df"]["gross_r"]


def min_equity(W, min_order=10.0):
    a = W.abs().where(W.abs() > 1e-9)
    mn = a.min(axis=1).dropna()
    if len(mn) == 0:
        return np.nan, np.nan
    eq = min_order / mn
    return float(eq.median()), float(eq.quantile(0.9))


def evaluate(W, name, U=None, rebalance_every=1, n_trials=None, full=True, extra=None):
    """Full candidate evaluation with HYPE cost. Returns flat dict."""
    res = lab.backtest(W, "1d", cost=COST, rebalance_every=rebalance_every)
    m = res["metrics"]
    r = res["df"]["r"]
    out = {"name": name, "sharpe": m.get("sharpe"), "cagr": m.get("cagr"), "maxdd": m.get("maxdd"),
           "t_nw": m.get("t_nw"), "vol": m.get("vol"), "turnover_per_year": m.get("turnover_per_year"),
           "cost_drag_per_year": m.get("cost_drag_per_year"), "funding_per_year": m.get("funding_per_year"),
           "avg_gross": m.get("avg_gross"), "avg_net": m.get("avg_net"), "avg_n_pos": m.get("avg_n_pos"),
           "skew": m.get("skew"), "kurt": m.get("kurt"), "days": m.get("days"), "start": m.get("start")}
    out.update({f"sh_{k}": v for k, v in sub_sharpes(r).items()})
    if full:
        r2 = lab.backtest(W, "1d", cost=2 * COST, rebalance_every=rebalance_every)["df"]["r"]
        rl = lab.backtest(W, "1d", cost=COST, extra_lag=1, rebalance_every=rebalance_every)["df"]["r"]
        out["sharpe_cost2x"] = sharpe(r2)
        out["sharpe_lag1"] = sharpe(rl)
        rL = lab.backtest(W.clip(lower=0), "1d", cost=COST, rebalance_every=rebalance_every)
        rS = lab.backtest(W.clip(upper=0), "1d", cost=COST, rebalance_every=rebalance_every)
        out["long_leg_sharpe"] = rL["metrics"].get("sharpe", np.nan) if rL["metrics"] else np.nan
        out["short_leg_sharpe"] = rS["metrics"].get("sharpe", np.nan) if rS["metrics"] else np.nan
        if U is not None:
            ew = ew_bench(U)
            for leg, rr, sgn in (("long", rL, 1), ("short", rS, -1)):
                if not rr["metrics"]:
                    out[f"{leg}_leg_active_sharpe"] = np.nan
                    continue
                d = rr["df"]
                act = d["r"] - sgn * d["gross_exposure"] * ew.reindex(d.index).fillna(0.0)
                out[f"{leg}_leg_active_sharpe"] = sharpe(act)
        out.update(alpha_vs_controls(r))
        me, me90 = min_equity(W)
        out["min_equity_median"] = me; out["min_equity_p90"] = me90
        y = lab.yearly(r, "1d")
        out["yearly"] = "; ".join(f"{i}: {v:+.1%} (SR {s:.2f})" for i, v, s in zip(y.index, y["ret"], y["sharpe"]))
        if n_trials:
            out["dsr"] = lab.deflated_sharpe(out["sharpe"], n_trials, out["days"], out["skew"], out["kurt"] + 3)
    if extra:
        out.update(extra)
    return out, res


def log(row, path=RUNLOG):
    row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
    exists = os.path.exists(path)
    keys = list(row.keys())
    if exists:
        with open(path, newline="", encoding="utf-8") as f:
            hdr = next(csv.reader(f), None)
        if hdr is not None and hdr != keys:
            # append with union header via pandas
            old = pd.read_csv(path)
            new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
            new.to_csv(path, index=False)
            return
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if not exists:
            w.writeheader()
        w.writerow(row)


def fmt(o, keys=("sharpe", "cagr", "maxdd", "sh_2020-21", "sh_2022", "sh_2023-24", "sharpe_cost2x",
                 "sharpe_lag1", "long_leg_sharpe", "short_leg_sharpe", "corr_btc", "corr_xsmom",
                 "alpha_t_vs_xsmom", "avg_n_pos", "turnover_per_year")):
    parts = [f"{o['name']:<44s}"]
    for k in keys:
        v = o.get(k)
        if v is None:
            continue
        parts.append(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}")
    return " ".join(parts)
