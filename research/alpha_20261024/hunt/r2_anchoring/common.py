"""Helpers for r2_anchoring (research period 2020-2024 only, via lab.py; OHLCV-only signals).

Signal inputs allowed: open high low close qv trades. Funding only enters as a COST via lab.backtest.
"""
import sys
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab
import numpy as np, pandas as pd

COST = lab.HYPE_COST
PERIODS = [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"),
           ("2023-24", "2023-01-01", "2025-01-01")]

_cache = {}


def C():
    return lab.load("close", "1d")


def R():
    if "R" not in _cache:
        _cache["R"] = C().pct_change(fill_method=None)
    return _cache["R"]


def U(top_k=50, hype=True):
    key = ("U", top_k, hype)
    if key not in _cache:
        _cache[key] = lab.hype_universe("1d", top_k=top_k) if hype else lab.universe("1d", top_k=top_k)
    return _cache[key]


def sh(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) > 20 and x.std() > 0 else np.nan


def subperiods(r):
    out = {}
    for name, a, b in PERIODS:
        x = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]
        out[name] = sh(x)
    return out


def yearly_str(r):
    y = lab.yearly(r, "1d")
    return ", ".join(f"{i}: {v:+.1%} (SR {s:.2f})" for i, v, s in zip(y.index, y["ret"], y["sharpe"]))


# ---------------------------------------------------------------- controls
def controls():
    """lab.controls (btc, xsmom14 on full top-50, 0.07%) + plain momentum L/S on the HYPE top-50
    universe (quintile, daily rebal, HYPE cost) for spanning tests."""
    if "ctrl" in _cache:
        return _cache["ctrl"]
    ct = lab.controls()
    c = C()
    u = U(50)
    for L in (14, 30, 60, 90, 180, 365):
        mom = c / c.shift(L) - 1
        ct[f"hmom{L}"] = lab.backtest(lab.quantile_ls_weights(mom, u, q=0.2), "1d", cost=COST)["df"]["r"]
    _cache["ctrl"] = ct
    return ct


def mom_ctrl(L, skip, rebal=1, u=None):
    """plain momentum L/S (same lookback/skip) on the HYPE universe: the 'matched momentum' control."""
    c = C()
    u = U(50) if u is None else u
    mom = c.shift(skip) / c.shift(skip + L) - 1
    return lab.backtest(lab.quantile_ls_weights(mom, u, q=0.2), "1d", cost=COST,
                        rebalance_every=rebal)["df"]["r"]


def ols_nw(y, X, lags=10):
    """OLS with Newey-West SE. Returns (coef, t) arrays, first = intercept."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
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
    return b, b / np.sqrt(np.diag(V))


def spanning(r, ctrl, cols):
    d = pd.concat([r.rename("y"), ctrl[cols]], axis=1).dropna()
    b, t = ols_nw(d["y"].values, d[cols].values)
    return {"alpha_ann": float(b[0] * 365), "alpha_t": float(t[0]),
            **{f"b_{c}": float(bb) for c, bb in zip(cols, b[1:])}, "n": len(d)}


def corr(r, s):
    d = pd.concat([r, s], axis=1).dropna()
    return float(d.corr().iloc[0, 1])


# ---------------------------------------------------------------- weights
def topn_weights(score, mask, n=8, gross=1.0):
    """n long (highest score) / n short (lowest), equal weight, each side gross/2.
    If fewer than 2n valid names, uses floor(valid/2) per side."""
    s = score.where(mask)
    cnt = s.notna().sum(axis=1)
    k = np.minimum(n, (cnt // 2)).astype(float)
    rk_hi = s.rank(axis=1, ascending=False, method="first")
    rk_lo = s.rank(axis=1, ascending=True, method="first")
    lo = rk_hi.le(k, axis=0) & s.notna()
    shm = rk_lo.le(k, axis=0) & s.notna()
    kk = k.replace(0, np.nan)
    W = lo.astype(float).div(kk, axis=0) - shm.astype(float).div(kk, axis=0)
    return W.fillna(0.0) * gross / 2


def beta_neutral(W, beta, lo=0.25, hi=0.75):
    """Rescale long and short legs (gross sum kept) so that long_gross*beta_L = short_gross*beta_S."""
    b = beta.clip(0.1, 3.0).reindex_like(W)
    Wl = W.clip(lower=0)
    Ws = -W.clip(upper=0)
    gl = Wl.sum(axis=1)
    gs = Ws.sum(axis=1)
    bl = (Wl * b).sum(axis=1) / gl.replace(0, np.nan)
    bs = (Ws * b).sum(axis=1) / gs.replace(0, np.nan)
    G = gl + gs
    fl = (bs / (bl + bs)).clip(lo, hi).fillna(0.5)
    Wn = Wl.div(gl.replace(0, np.nan), axis=0).mul(fl * G, axis=0) - \
        Ws.div(gs.replace(0, np.nan), axis=0).mul((1 - fl) * G, axis=0)
    return Wn.fillna(0.0)


def rolling_beta(B=90, minp=60):
    """trailing beta of each coin vs BTC over B days ending at t (known at close t)."""
    if ("beta", B) in _cache:
        return _cache[("beta", B)]
    r = R()
    rb = r["BTCUSDT"]
    rbm = pd.DataFrame(np.repeat(rb.values[:, None], r.shape[1], axis=1), index=r.index, columns=r.columns)
    rbm = rbm.where(r.notna())
    rr = r.where(rbm.notna())
    mxy = (rr * rbm).rolling(B, min_periods=minp).mean()
    mx = rr.rolling(B, min_periods=minp).mean()
    my = rbm.rolling(B, min_periods=minp).mean()
    vy = (rbm ** 2).rolling(B, min_periods=minp).mean() - my ** 2
    beta = (mxy - mx * my) / vy
    _cache[("beta", B)] = beta
    return beta


# ---------------------------------------------------------------- evaluation
def run(W, rebal=1, cost=COST, extra_lag=0):
    return lab.backtest(W, "1d", cost=cost, rebalance_every=rebal, extra_lag=extra_lag)


def ew_univ(u):
    Rn = R().shift(-1)
    ew = Rn.where(u & Rn.notna()).mean(axis=1)
    ew.index = ew.index + pd.Timedelta("1D")
    return ew


def min_equity(W):
    a = W.abs().where(W.abs() > 1e-9)
    mn = a.min(axis=1).dropna()
    eq = 10.0 / mn
    return float(eq.median()), float(eq.quantile(0.9))


def full_eval(W, name, u, rebal=1, n_trials=1, legs=True, ctrl=None, mom_match=None):
    res = run(W, rebal)
    df, m = res["df"], res["metrics"]
    r = df["r"]
    out = {"name": name, "sharpe": m["sharpe"], "cagr": m["cagr"], "maxdd": m["maxdd"], "t_nw": m["t_nw"],
           "turn": m["turnover_per_year"], "cost_yr": m["cost_drag_per_year"], "fund_yr": m["funding_per_year"],
           "npos": m["avg_n_pos"], "skew": m["skew"], "kurt": m["kurt"], "days": m["days"]}
    for k, v in subperiods(r).items():
        out["sh_" + k] = v
    out["sh_cost2x"] = run(W, rebal, cost=2 * COST)["metrics"]["sharpe"]
    out["sh_lag1"] = run(W, rebal, extra_lag=1)["metrics"]["sharpe"]
    out["dsr"] = lab.deflated_sharpe(m["sharpe"], n_trials, m["days"], m["skew"], m["kurt"] + 3)
    out["mineq_med"], out["mineq_p90"] = min_equity(W)
    out["yearly"] = yearly_str(r)
    if legs:
        ew = ew_univ(u)
        for side, Wl in [("long", W.clip(lower=0)), ("short", W.clip(upper=0))]:
            rl = run(Wl * 2, rebal)["df"]["r"]
            out["sh_" + side] = sh(rl)
            e = ew.reindex(rl.index).fillna(0)
            out["sh_" + side + "_vsEW"] = sh(rl - e) if side == "long" else sh(rl + e)
    if ctrl is not None:
        out["corr_btc"] = corr(r, ctrl["btc"])
        out["corr_xsmom14"] = corr(r, ctrl["xsmom14"])
        out["corr_hmom90"] = corr(r, ctrl["hmom90"])
        s1 = spanning(r, ctrl, ["xsmom14", "btc"])
        out["alpha_t_xsmom14_btc"] = s1["alpha_t"]
        out["alpha_ann_xsmom14_btc"] = s1["alpha_ann"]
        s2 = spanning(r, ctrl, ["hmom90", "btc"])
        out["alpha_t_mom90_btc"] = s2["alpha_t"]
        out["b_mom90"] = s2["b_hmom90"]
        s3 = spanning(r, ctrl, ["xsmom14", "hmom30", "hmom60", "hmom90", "hmom180", "btc"])
        out["alpha_t_allmom"] = s3["alpha_t"]
        out["alpha_ann_allmom"] = s3["alpha_ann"]
        if mom_match is not None:
            cc = ctrl.copy()
            cc["mm"] = mom_match
            s4 = spanning(r, cc, ["mm", "btc"])
            out["alpha_t_matched_mom"] = s4["alpha_t"]
            out["corr_matched_mom"] = corr(r, mom_match)
    return out, res


def quick(W, rebal=1):
    """grid-stage metrics: net Sharpe, subperiods, turnover, n_pos."""
    res = run(W, rebal)
    m = res["metrics"]
    r = res["df"]["r"]
    o = {"sharpe": m["sharpe"], "cagr": m["cagr"], "maxdd": m["maxdd"], "t_nw": m["t_nw"],
         "turn": m["turnover_per_year"], "fund_yr": m["funding_per_year"], "npos": m["avg_n_pos"]}
    for k, v in subperiods(r).items():
        o["sh_" + k] = v
    return o, r
