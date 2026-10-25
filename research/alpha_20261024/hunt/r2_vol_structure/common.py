"""Shared helpers for hunter r2_vol_structure (round 2).

Data: ONLY via lab.py (IS 2020-01-01..2024-12-31). ALPHA_FORWARD is never set here.
Signal inputs: only open/high/low/close (and qv via lab's universe ranking) - HYPE-candle fields.
Funding is charged as a COST inside lab.backtest (not used as a signal).
"""
import os, sys
import numpy as np, pandas as pd

ROOT = r"C:\Crypto data\backtest data and more\alpha_20261024"
sys.path.insert(0, ROOT)
import lab  # noqa: E402

assert os.environ.get("ALPHA_FORWARD") != "1", "forward data must never be enabled by a hunter"
HERE = os.path.dirname(os.path.abspath(__file__))
COST = lab.HYPE_COST
BTC = "BTCUSDT"
SUBS = [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"),
        ("2023-24", "2023-01-01", "2025-01-01")]


# ------------------------------------------------------------------ data
def returns():
    return lab.load("close", "1d").astype("float64").pct_change(fill_method=None)


def universe(k=50, ex_btc=False):
    U = lab.hype_universe("1d", top_k=k)
    if ex_btc:
        U = U.copy(); U[BTC] = False
    return U


_ctrl = None


def controls():
    """lab.controls() (BTC daily return, xsmom14 top-50 quintile L/S net), cached in this folder."""
    global _ctrl
    if _ctrl is None:
        p = os.path.join(HERE, "_controls_IS.parquet")
        if os.path.exists(p):
            _ctrl = pd.read_parquet(p)
        else:
            _ctrl = lab.controls()
            _ctrl.to_parquet(p)
        _ctrl = _ctrl[_ctrl.index < lab.IS_END]
    return _ctrl


# ------------------------------------------------------------------ weights
def _rank_sel(score, mask, q=0.2, n=None):
    s = score.where(mask)
    if n is None:
        r = s.rank(axis=1, pct=True)
        lo = r >= 1 - q
        sh = r <= q
    else:
        cnt = s.notna().sum(axis=1)
        rd = s.rank(axis=1, ascending=False, method="first")
        ra = s.rank(axis=1, ascending=True, method="first")
        nn = np.minimum(n, (cnt // 2)).values[:, None]   # never overlap
        lo = rd.values <= nn
        sh = ra.values <= nn
        lo = pd.DataFrame(lo, index=s.index, columns=s.columns) & s.notna()
        sh = pd.DataFrame(sh, index=s.index, columns=s.columns) & s.notna()
    return lo.astype(float), sh.astype(float)


def ls_weights(score, mask, q=0.2, n=None, gross=1.0):
    """$-neutral: long top-q (or top-n), short bottom-q (or bottom-n), equal weight, each side gross/2."""
    lo, sh = _rank_sel(score, mask, q, n)
    nl = lo.sum(axis=1).replace(0, np.nan); ns = sh.sum(axis=1).replace(0, np.nan)
    W = lo.div(nl, axis=0) - sh.div(ns, axis=0)
    ok = (nl.notna() & ns.notna())
    return W.where(ok, 0.0).fillna(0.0) * gross / 2


def bn_weights(score, mask, beta, q=0.2, n=None, gross=1.0, beta_floor=0.25):
    """Beta-neutral legs: EW within leg, leg gross g_L ~ 1/beta_L, g_S ~ 1/beta_S, g_L+g_S = gross.
    -> ex-ante beta exposure 0 (beta measured on trailing window, known at t)."""
    lo, sh = _rank_sel(score, mask, q, n)
    b = beta.clip(lower=beta_floor)
    nl = lo.sum(axis=1).replace(0, np.nan); ns = sh.sum(axis=1).replace(0, np.nan)
    bL = (lo * b).sum(axis=1) / nl
    bS = (sh * b).sum(axis=1) / ns
    gL = (1 / bL) / (1 / bL + 1 / bS) * gross
    gS = (1 / bS) / (1 / bL + 1 / bS) * gross
    W = lo.div(nl, axis=0).mul(gL, axis=0) - sh.div(ns, axis=0).mul(gS, axis=0)
    ok = nl.notna() & ns.notna() & bL.notna() & bS.notna()
    return W.where(ok, 0.0).fillna(0.0)


def ew_weights(mask):
    m = mask.astype(float)
    return m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


# ------------------------------------------------------------------ stats
def sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(365)) if len(r) > 10 and r.std() > 0 else np.nan


def sub_sharpes(r):
    return {nm: sharpe(r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))])
            for nm, a, b in SUBS}


def ols_nw(y, X, lags=10):
    """OLS with Newey-West HAC SE. y Series, X DataFrame (no const). returns coef, t (const first)."""
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    Y = d["y"].values; Z = np.column_stack([np.ones(len(d)), d[X.columns].values])
    XtX_inv = np.linalg.inv(Z.T @ Z)
    b = XtX_inv @ Z.T @ Y
    e = Y - Z @ b
    n = len(Y)
    S = (Z * e[:, None]).T @ (Z * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (Z[L:] * e[L:, None]).T @ (Z[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    names = ["const"] + list(X.columns)
    return dict(zip(names, b)), dict(zip(names, b / se)), n


def min_equity(W):
    """$ equity needed so every open position is >= $10 (median and p90 over active days)."""
    A = W.abs().where(W.abs() > 1e-9)
    mn = A.min(axis=1).dropna()
    if len(mn) == 0:
        return np.nan, np.nan
    me = 10.0 / mn
    return float(me.median()), float(me.quantile(0.9))


def evaluate(W, name, rebalance=1, extra=True, mask=None, n_trials=None):
    """Full metric set for one weight panel. W are TARGET weights decided at close t."""
    res = lab.backtest(W, "1d", cost=COST, rebalance_every=rebalance)
    m = res["metrics"]; df = res["df"]; r = df["r"]
    out = {"name": name, "sharpe": m["sharpe"], "cagr": m["cagr"], "vol": m["vol"], "maxdd": m["maxdd"],
           "t_nw": m["t_nw"], "turnover_per_year": m["turnover_per_year"],
           "cost_drag_per_year": m["cost_drag_per_year"], "funding_per_year": m["funding_per_year"],
           "avg_gross": m["avg_gross"], "avg_net": m["avg_net"], "avg_n_pos": m["avg_n_pos"],
           "gross_sharpe": sharpe(df["gross_r"]), "skew": m["skew"], "kurt": m["kurt"], "days": m["days"],
           "start": m["start"]}
    out.update({f"sh_{k}": v for k, v in sub_sharpes(r).items()})
    yr = lab.yearly(r, "1d")
    out["yearly"] = " ".join(f"{y}:{v:+.0%}" for y, v in yr["ret"].items())
    if not extra:
        return out, res
    out["sharpe_cost2x"] = lab.backtest(W, "1d", cost=2 * COST, rebalance_every=rebalance)["metrics"]["sharpe"]
    out["sharpe_lag1"] = lab.backtest(W, "1d", cost=COST, rebalance_every=rebalance, extra_lag=1)["metrics"]["sharpe"]
    # controls
    C = controls()
    j = pd.concat([r.rename("s"), C], axis=1).dropna()
    out["corr_btc"] = float(j["s"].corr(j["btc"])); out["corr_xsmom"] = float(j["s"].corr(j["xsmom14"]))
    b, t, _ = ols_nw(j["s"], j[["xsmom14", "btc"]])
    out["alpha_ann"] = b["const"] * 365; out["alpha_t"] = t["const"]
    out["beta_xsmom"] = b["xsmom14"]; out["beta_btc"] = b["btc"]
    # legs (raw, at the book's own leg gross) and leg excess vs EW universe (each leg scaled to gross 1)
    WL = W.clip(lower=0); WS = W.clip(upper=0)
    rl = lab.backtest(WL, "1d", cost=COST, rebalance_every=rebalance)["df"]["r"]
    rs = lab.backtest(WS, "1d", cost=COST, rebalance_every=rebalance)["df"]["r"]
    out["long_leg_sharpe_raw"] = sharpe(rl); out["short_leg_sharpe_raw"] = sharpe(rs)
    if mask is not None:
        rew = lab.backtest(ew_weights(mask), "1d", cost=COST, rebalance_every=rebalance)["df"]["r"]
        gl = WL.sum(axis=1).replace(0, np.nan); gs = (-WS).sum(axis=1).replace(0, np.nan)
        rl1 = lab.backtest(WL.div(gl, axis=0).fillna(0), "1d", cost=COST, rebalance_every=rebalance)["df"]["r"]
        rs1 = lab.backtest((-WS).div(gs, axis=0).fillna(0), "1d", cost=COST, rebalance_every=rebalance)["df"]["r"]
        jj = pd.concat([rl1.rename("l"), rs1.rename("s"), rew.rename("e")], axis=1).dropna()
        out["long_leg_xs_sharpe"] = sharpe(jj["l"] - jj["e"]); out["short_leg_xs_sharpe"] = sharpe(jj["e"] - jj["s"])
    # sizing (on rebalance-sampled targets)
    Wt = W
    if rebalance > 1:
        keep = np.arange(len(W)) % rebalance == 0
        Wt = W.where(pd.Series(keep, index=W.index), np.nan).ffill().fillna(0.0)
    out["min_eq_med"], out["min_eq_p90"] = min_equity(Wt)
    if n_trials:
        out["dsr"] = lab.deflated_sharpe(m["sharpe"], n_trials, m["days"], m["skew"], m["kurt"] + 3)
    return out, res


def fmt_row(o, keys=("sharpe", "gross_sharpe", "sh_2020-21", "sh_2022", "sh_2023-24", "turnover_per_year",
                     "funding_per_year", "avg_n_pos")):
    return "  ".join(f"{k}={o.get(k, np.nan):.2f}" for k in keys)
