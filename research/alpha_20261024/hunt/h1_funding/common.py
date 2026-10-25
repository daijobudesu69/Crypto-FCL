"""Helpers for h1_funding (research period 2020-2024 only, via lab.py)."""
import sys
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab
import numpy as np, pandas as pd

_ctrl = None


def controls():
    global _ctrl
    if _ctrl is None:
        _ctrl = lab.controls()
    return _ctrl


def sh(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) > 20 and x.std() > 0 else np.nan


def subperiods(r):
    out = {}
    for lab_, a, b in [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"),
                       ("2023-24", "2023-01-01", "2025-01-01")]:
        x = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]
        out[lab_] = sh(x)
    return out


def ew_univ(U):
    """equal-weight next-bar return of universe (no costs), aligned to backtest df index."""
    R = lab.returns("1d")
    Rn = R.shift(-1)
    m = U & Rn.notna()
    ew = Rn.where(m).mean(axis=1)
    ew.index = ew.index + pd.Timedelta("1D")
    return ew


def row(W, name, U=None, cost=lab.COST, rebalance_every=1, extra_lag=0, legs=True):
    res = lab.backtest(W, "1d", cost=cost, rebalance_every=rebalance_every, extra_lag=extra_lag)
    df, m = res["df"], res["metrics"]
    d = {"name": name, "sharpe": m["sharpe"], "cagr": m["cagr"], "maxdd": m["maxdd"], "t_nw": m["t_nw"],
         "turn": m["turnover_per_year"], "cost_yr": m["cost_drag_per_year"], "fund_yr": m["funding_per_year"],
         "gross_yr": float(df["gross_r"].mean() * 365), "npos": m["avg_n_pos"],
         "sh_price_net": sh(df["gross_r"] - df["cost_r"]), "sh_price_gross": sh(df["gross_r"]),
         "sh_fund": sh(df["fund_r"]), "skew": m["skew"], "kurt": m["kurt"], "days": m["days"]}
    for k, v in subperiods(df["r"]).items():
        d["sh_" + k] = v
    if legs and U is not None:
        ew = ew_univ(U)
        for side, Wl in [("long", W.clip(lower=0)), ("short", W.clip(upper=0))]:
            rl = lab.backtest(Wl * 2, "1d", cost=cost, rebalance_every=rebalance_every)["df"]["r"]
            d["sh_" + side] = sh(rl)
            e = ew.reindex(rl.index).fillna(0)
            d["sh_" + side + "_vsEW"] = sh(rl - e) if side == "long" else sh(rl + e)
    return d, res


def stress(W, rebalance_every=1):
    a = lab.backtest(W, "1d", cost=2 * lab.COST, rebalance_every=rebalance_every)["metrics"]["sharpe"]
    b = lab.backtest(W, "1d", extra_lag=1, rebalance_every=rebalance_every)["metrics"]["sharpe"]
    return a, b


def corr_ctrl(r):
    c = controls().reindex(r.index)
    return float(r.corr(c["btc"])), float(r.corr(c["xsmom14"]))


def min_equity(W, min_order=10.0, q=0.1):
    a = W.abs().where(W.abs() > 1e-9)
    mn = a.min(axis=1).dropna()
    mn = mn[mn > 0]
    return float(min_order / mn.quantile(q)) if len(mn) else np.nan


def yearly_decomp(res):
    df = res["df"]
    g = df.groupby(df.index.year)
    return pd.DataFrame({"net": g["r"].apply(lambda x: (1 + x).prod() - 1),
                         "price_sum": g["gross_r"].sum(), "fund_sum": g["fund_r"].sum(), "cost_sum": g["cost_r"].sum(),
                         "sharpe": g["r"].apply(sh)})
