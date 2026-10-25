"""h3_basis_flow helper: one evaluation routine used by every config so all are judged alike.

evaluate(W, name, family, tf) runs:
  base backtest (0.07%/side + real funding), 2x cost, extra_lag=1, long-leg only, short-leg only
  and returns a flat dict of metrics; every call is appended to registry.csv (config counting
  for the deflated Sharpe). Research data 2020-2024 only (lab.py truncates; ALPHA_FORWARD never set).
"""
import os, sys, json
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import numpy as np, pandas as pd
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "registry.csv")
_ctrl = None
_ew = {}


def ew_ret(tf="1d"):
    """EW top-50 universe basket net daily return (benchmark for leg-excess Sharpe)."""
    if tf not in _ew:
        U = lab.universe(tf, min_hist_days=60, top_k=50)
        W = U.astype(float).div(U.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        _ew[tf] = lab.backtest(W, tf)["df"]["r"]
    return _ew[tf]


def controls():
    global _ctrl
    if _ctrl is None:
        p = os.path.join(HERE, "_controls.parquet")
        if os.path.exists(p):
            _ctrl = pd.read_parquet(p)
        else:
            _ctrl = lab.controls()
            _ctrl.to_parquet(p)
    return _ctrl


def sh(d):
    d = d.dropna()
    return float(d.mean() / d.std() * np.sqrt(365)) if len(d) > 20 and d.std() > 0 else np.nan


def subperiods(d):
    out = {}
    for k, (a, b) in {"2020-21": ("2020-01-01", "2022-01-01"), "2022": ("2022-01-01", "2023-01-01"),
                      "2023-24": ("2023-01-01", "2025-01-01")}.items():
        x = d[(d.index >= pd.Timestamp(a, tz="UTC")) & (d.index < pd.Timestamp(b, tz="UTC"))]
        out[k] = sh(x)
    return out


def legs(W):
    Wl = W.clip(lower=0)
    Ws = W.clip(upper=0)
    return Wl, Ws


def min_equity(W):
    a = W.abs().where(W.abs() > 1e-9)
    m = a.min(axis=1).dropna()
    if len(m) == 0:
        return np.nan
    return float(np.nanpercentile(10.0 / m, 95))


def evaluate(W, name, family, tf="1d", note="", save=True, rebalance_every=1, daily_extra=False):
    W = W.fillna(0.0)
    res = lab.backtest(W, tf, rebalance_every=rebalance_every)
    m = res["metrics"]
    df = res["df"]
    d = lab.to_daily(df["r"], tf)
    r2 = lab.backtest(W, tf, cost=2 * lab.COST, rebalance_every=rebalance_every)["metrics"]["sharpe"]
    rl = lab.backtest(W, tf, extra_lag=1, rebalance_every=rebalance_every)["metrics"]["sharpe"]
    Wl, Ws = legs(W)
    ml = lab.backtest(Wl, tf, rebalance_every=rebalance_every)["metrics"] if (Wl.values > 0).any() else {}
    ms = lab.backtest(Ws, tf, rebalance_every=rebalance_every)["metrics"] if (Ws.values < 0).any() else {}
    ew = ew_ret(tf)
    lx = sx = np.nan
    if ml:
        dl = lab.backtest(Wl, tf, rebalance_every=rebalance_every)["df"]
        e = ew.reindex(dl.index).fillna(0.0)
        lx = sh(lab.to_daily(dl["r"] - dl["gross_exposure"] * e, tf))
    if ms:
        ds = lab.backtest(Ws, tf, rebalance_every=rebalance_every)["df"]
        e = ew.reindex(ds.index).fillna(0.0)
        sx = sh(lab.to_daily(ds["r"] + ds["gross_exposure"] * e, tf))
    c = controls()
    j = pd.concat([d.rename("s"), c], axis=1, sort=True).dropna()
    sp = subperiods(d)
    yr = lab.yearly(df["r"], tf)
    bpy = lab.BARS_PER_YEAR[tf]
    out = {"family": family, "name": name, "tf": tf, "note": note,
           "start": m["start"], "end": m["end"], "sharpe": m["sharpe"], "cagr": m["cagr"], "vol": m["vol"],
           "maxdd": m["maxdd"], "t_nw": m["t_nw"], "skew": m["skew"], "kurt": m["kurt"],
           "sharpe_2x": r2, "sharpe_lag1": rl,
           "sh_2020_21": sp["2020-21"], "sh_2022": sp["2022"], "sh_2023_24": sp["2023-24"],
           "long_sharpe": ml.get("sharpe", np.nan), "short_sharpe": ms.get("sharpe", np.nan),
           "long_ex_sharpe": lx, "short_ex_sharpe": sx,
           "price_per_year": float(df["gross_r"].mean() * bpy), "funding_per_year": m["funding_per_year"],
           "cost_per_year": m["cost_drag_per_year"], "turnover_per_year": m["turnover_per_year"],
           "avg_gross": m["avg_gross"], "avg_net": m["avg_net"], "avg_n_pos": m["avg_n_pos"],
           "corr_btc": float(j["s"].corr(j["btc"])) if len(j) else np.nan,
           "corr_xsmom": float(j["s"].corr(j["xsmom14"])) if len(j) else np.nan,
           "min_equity": min_equity(W),
           "yearly": json.dumps({int(k): round(float(v), 4) for k, v in yr["ret"].items()}),
           "yearly_sh": json.dumps({int(k): round(float(v), 2) for k, v in yr["sharpe"].items()})}
    if save:
        hdr = not os.path.exists(REG)
        pd.DataFrame([out]).to_csv(REG, mode="a", header=hdr, index=False)
    return out, res


def fmt(o):
    return (f"{o['name']:38s} Sh {o['sharpe']:+.2f} 2x {o['sharpe_2x']:+.2f} lag1 {o['sharpe_lag1']:+.2f} | "
            f"20-21 {o['sh_2020_21']:+.2f} 22 {o['sh_2022']:+.2f} 23-24 {o['sh_2023_24']:+.2f} | "
            f"L {o['long_sharpe']:+.2f} S {o['short_sharpe']:+.2f} Lx {o['long_ex_sharpe']:+.2f} Sx {o['short_ex_sharpe']:+.2f} | px {o['price_per_year']:+.1%} "
            f"fund {o['funding_per_year']:+.1%} cost {o['cost_per_year']:.1%} | cBTC {o['corr_btc']:+.2f} "
            f"cMOM {o['corr_xsmom']:+.2f} | dd {o['maxdd']:.0%}")


def rank_ic(score, fwd, mask):
    """Mean daily Spearman cross-sectional IC between score[t] and fwd[t] (fwd must be future return
    aligned at t, i.e. built with shift(-h)). Diagnostic only."""
    s = score.where(mask).rank(axis=1)
    f = fwd.where(mask & score.notna()).rank(axis=1)
    s = s.where(f.notna())
    sc = s.sub(s.mean(axis=1), axis=0); fc = f.sub(f.mean(axis=1), axis=0)
    ic = (sc * fc).sum(axis=1) / np.sqrt((sc ** 2).sum(axis=1) * (fc ** 2).sum(axis=1))
    ic = ic[(s.notna().sum(axis=1) >= 8)]
    return ic


def ts_z(x, win=90, minp=30):
    """Trailing (no lookahead) z-score per column."""
    m = x.rolling(win, min_periods=minp).mean()
    s = x.rolling(win, min_periods=minp).std()
    return (x - m) / s.replace(0, np.nan)
