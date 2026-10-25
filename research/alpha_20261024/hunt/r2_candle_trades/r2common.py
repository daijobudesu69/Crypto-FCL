"""Shared helpers for hunter r2_candle_trades (round 2: HYPE-candle-only signals).

Rules obeyed: data only via lab.py (truncated at 2025-01-01), ALPHA_FORWARD never set, signal inputs
limited to open/high/low/close/qv/trades, all rolling stats trailing, cost = lab.HYPE_COST, funding charged
as cost by lab.backtest.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COST = lab.HYPE_COST
MIN_NAMES = 10          # pre-registered: no position on days with < 10 eligible names
SUBP = [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"),
        ("2023-24", "2023-01-01", "2025-01-01")]
_cache = {}


def panels():
    if "p" not in _cache:
        O = lab.load("open", "1d"); H = lab.load("high", "1d"); L = lab.load("low", "1d")
        C = lab.load("close", "1d"); Q = lab.load("qv", "1d"); T = lab.load("trades", "1d")
        bad = (H - L <= 0) | (T <= 0) | (Q <= 0)
        _cache["p"] = dict(O=O.where(~bad), H=H.where(~bad), L=L.where(~bad), C=C, Q=Q.where(~bad),
                           T=T.where(~bad))
    return _cache["p"]


def U(top_k=50):
    k = ("U", top_k)
    if k not in _cache:
        u = lab.hype_universe("1d", top_k=top_k)
        n = u.sum(axis=1)
        _cache[k] = u & (n >= MIN_NAMES).values[:, None]
    return _cache[k]


def ctrl():
    if "ctrl" not in _cache:
        _cache["ctrl"] = lab.controls()
    return _cache["ctrl"]


def mom(n):
    C = panels()["C"]
    return C / C.shift(n) - 1


# ------------------------------------------------------------------ cross-sectional tools
def xs_resid(y, xs, mask):
    """Daily cross-sectional OLS residual of rank(y) on ranks of the regressors xs (list), within mask.
    Uses only same-day information -> no lookahead."""
    m = mask & y.notna()
    for x in xs:
        m &= x.notna()
    ry = lab.xs_rank(y, m).values
    rx = [lab.xs_rank(x, m).values for x in xs]
    out = np.full(ry.shape, np.nan)
    for t in range(ry.shape[0]):
        ok = ~np.isnan(ry[t])
        if ok.sum() < 8:
            continue
        X = np.column_stack([np.ones(ok.sum())] + [r[t, ok] for r in rx])
        b, *_ = np.linalg.lstsq(X, ry[t, ok], rcond=None)
        out[t, ok] = ry[t, ok] - X @ b
    return pd.DataFrame(out, index=y.index, columns=y.columns)


def topn_weights(score, mask, n=8, gross=1.0):
    """Long n highest, short n lowest, equal weight, gross split 50/50. Needs >= 2n names."""
    s = score.where(mask)
    cnt = s.notna().sum(axis=1)
    r_hi = s.rank(axis=1, ascending=False, method="first")
    r_lo = s.rank(axis=1, ascending=True, method="first")
    ok = (cnt >= 2 * n).values[:, None]
    lo = ((r_hi <= n) & ok).astype(float)
    sh = ((r_lo <= n) & ok).astype(float)
    return (lo - sh) * gross / (2 * n)


def qls(score, mask, q=0.2):
    return lab.quantile_ls_weights(score, mask, q=q)


# ------------------------------------------------------------------ statistics
def sharpe(d):
    d = d.dropna()
    return float(d.mean() / d.std() * np.sqrt(365)) if len(d) > 20 and d.std() > 0 else np.nan


def nw_ols(y, X, lags=10):
    """OLS with Newey-West HAC t-stats. X includes constant. Returns coef, t."""
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
    return b, b / np.sqrt(np.diag(V))


def alpha_vs_controls(d):
    c = ctrl()
    j = pd.concat([d.rename("s"), c], axis=1).dropna()
    X = np.column_stack([np.ones(len(j)), j["xsmom14"].values, j["btc"].values])
    b, t = nw_ols(j["s"].values, X)
    Xm = np.column_stack([np.ones(len(j)), j["xsmom14"].values])
    bm, tm = nw_ols(j["s"].values, Xm)
    return {"corr_btc": float(j["s"].corr(j["btc"])), "corr_xsmom": float(j["s"].corr(j["xsmom14"])),
            "alpha_ann_vs_mom_btc": float(b[0] * 365), "alpha_t_vs_mom_btc": float(t[0]),
            "beta_xsmom": float(b[1]), "beta_btc": float(b[2]),
            "alpha_t_vs_mom": float(tm[0]), "alpha_ann_vs_mom": float(bm[0] * 365)}


def subperiods(d):
    out = {}
    for name, a, b in SUBP:
        out["sh_" + name] = sharpe(d[(d.index >= pd.Timestamp(a, tz="UTC")) & (d.index < pd.Timestamp(b, tz="UTC"))])
    return out


def ew_next(Um):
    C = panels()["C"]
    R = C.pct_change(fill_method=None)
    Rn = R.shift(-1)
    ew = Rn.where(Um & Rn.notna()).mean(axis=1)
    ew.index = ew.index + pd.Timedelta("1D")
    return ew


def evaluate(W, name, family, log, Um=None, full=False, extra=None):
    res = lab.backtest(W, "1d", cost=COST)
    m = res["metrics"]
    df = res["df"]
    d = df["r"]
    row = {"family": family, "name": name}
    for k in ["start", "end", "days", "cagr", "vol", "sharpe", "maxdd", "t_nw", "skew", "kurt",
              "turnover_per_year", "cost_drag_per_year", "funding_per_year", "avg_gross", "avg_n_pos",
              "pct_pos_months"]:
        row[k] = m.get(k)
    row["sharpe_gross"] = sharpe(df["gross_r"])
    row.update(subperiods(d))
    row.update(alpha_vs_controls(d))
    yr = lab.yearly(d, "1d")
    row["yearly"] = "; ".join(f"{y}: {v:+.1%}" for y, v in yr["ret"].items())
    row["yearly_sharpe"] = "; ".join(f"{y}: {v:+.2f}" for y, v in yr["sharpe"].items())
    if full:
        row["sharpe_cost2x"] = lab.backtest(W, "1d", cost=2 * COST)["metrics"]["sharpe"]
        row["sharpe_lag1"] = lab.backtest(W, "1d", cost=COST, extra_lag=1)["metrics"]["sharpe"]
        rl = lab.backtest(W.clip(lower=0), "1d", cost=COST)
        rs = lab.backtest(W.clip(upper=0), "1d", cost=COST)
        row["long_leg_sharpe"] = rl["metrics"].get("sharpe")
        row["short_leg_sharpe"] = rs["metrics"].get("sharpe")
        if Um is not None:
            ew = ew_next(Um)
            Wf = W.reindex(index=panels()["C"].index, columns=panels()["C"].columns).fillna(0)
            gl = pd.Series(Wf.clip(lower=0).sum(axis=1).values, index=ew.index)
            gs = pd.Series((-Wf.clip(upper=0)).sum(axis=1).values, index=ew.index)
            lex = rl["df"]["r"] - (gl * ew).reindex(rl["df"].index).fillna(0)
            sex = rs["df"]["r"] + (gs * ew).reindex(rs["df"].index).fillna(0)
            row["long_excess_sharpe"] = sharpe(lex)
            row["short_excess_sharpe"] = sharpe(sex)
            row["long_excess_ann"] = float(lex.mean() * 365)
            row["short_excess_ann"] = float(sex.mean() * 365)
        Wa = W.reindex(index=df.index - pd.Timedelta("1D")).abs()
        mn = Wa.where(Wa > 1e-9).min(axis=1).dropna()
        row["min_equity_usd"] = float(10 / mn.median()) if len(mn) else np.nan
    if extra:
        row.update(extra)
    pd.DataFrame([row]).to_csv(log, mode="a", header=not os.path.exists(log), index=False)
    return row, res


def fmt(row):
    def g(k, f="{:.2f}"):
        v = row.get(k)
        return f.format(v) if v is not None and v == v else "nan"
    s = (f"{row['name']:34s} SR {g('sharpe')} (gr {g('sharpe_gross')}) | 20-21 {g('sh_2020-21')} 22 {g('sh_2022')} "
         f"23-24 {g('sh_2023-24')} | CAGR {g('cagr', '{:+.1%}')} DD {g('maxdd', '{:.0%}')} cost {g('cost_drag_per_year', '{:.1%}')} "
         f"fund {g('funding_per_year', '{:+.1%}')} | cMOM {g('corr_xsmom')} cBTC {g('corr_btc')} a_t {g('alpha_t_vs_mom_btc')} "
         f"npos {g('avg_n_pos', '{:.0f}')}")
    if "sharpe_cost2x" in row:
        s += (f"\n{'':34s} 2x {g('sharpe_cost2x')} lag1 {g('sharpe_lag1')} | Lraw {g('long_leg_sharpe')} Sraw {g('short_leg_sharpe')} "
              f"Lex {g('long_excess_sharpe')} Sex {g('short_excess_sharpe')} | minEq ${g('min_equity_usd', '{:.0f}')}")
    return s
