"""Shared helpers for hunter h4_xs_anomalies (short-term reversal + low-risk/lottery anomalies).

All data comes through lab.py (truncated at 2025-01-01). All rolling stats are trailing only.
"""
import os, sys, json
import numpy as np, pandas as pd

sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "grid_log.csv")
BTC = "BTCUSDT"
_ctrl = None


def controls():
    global _ctrl
    if _ctrl is None:
        _ctrl = lab.controls()
    return _ctrl


# ---------------------------------------------------------------- factor building blocks
def rolling_beta(R, mkt, win, minp=None):
    """Trailing OLS beta (and residual var, total var) of each column of R on mkt over win bars."""
    minp = minp or int(win * 2 / 3)
    M = pd.DataFrame(np.repeat(mkt.values[:, None], R.shape[1], axis=1), index=R.index, columns=R.columns)
    M = M.where(R.notna())
    Rm = R.where(M.notna())
    ex = Rm.rolling(win, min_periods=minp).mean()
    em = M.rolling(win, min_periods=minp).mean()
    exm = (Rm * M).rolling(win, min_periods=minp).mean()
    emm = (M * M).rolling(win, min_periods=minp).mean()
    exx = (Rm * Rm).rolling(win, min_periods=minp).mean()
    cov = exm - ex * em
    vm = emm - em * em
    vx = exx - ex * ex
    beta = cov / vm
    resvar = (vx - cov * cov / vm).clip(lower=0)
    return beta, resvar, vx


def shrink_beta(beta, w=0.6):
    """Vasicek-style shrink toward the cross-sectional median of the day (Frazzini-Pedersen use 0.6/0.4)."""
    med = beta.median(axis=1)
    return beta.mul(w).add(med * (1 - w), axis=0)


def ew_return(R, U):
    """Equal-weight next-bar return of universe U (benchmark for leg decomposition), aligned like backtest df."""
    Rn = R.shift(-1)
    m = U & Rn.notna()
    return Rn.where(m).mean(axis=1)


def beta_neutral_weights(score, U, beta, q=0.2, leg_beta=0.5, leg_cap=1.0):
    """Long top-q / short bottom-q (equal weight inside a leg); each leg scaled so its beta = leg_beta
    (Frazzini-Pedersen BAB style: low-beta leg levered up, high-beta leg delevered). Leg gross capped."""
    m = U & beta.notna() & score.notna()
    r = lab.xs_rank(score, m)
    lo = (r >= 1 - q).astype(float)
    sh = (r <= q).astype(float)
    nl = lo.sum(axis=1).replace(0, np.nan)
    ns = sh.sum(axis=1).replace(0, np.nan)
    wl = lo.div(nl, axis=0)
    ws = sh.div(ns, axis=0)
    bl = (wl * beta).sum(axis=1)
    bs = (ws * beta).sum(axis=1)
    kl = (leg_beta / bl.where(bl > 0.05)).clip(upper=leg_cap)
    ks = (leg_beta / bs.where(bs > 0.05)).clip(upper=leg_cap)
    W = wl.mul(kl, axis=0) - ws.mul(ks, axis=0)
    return W.fillna(0.0)


# ---------------------------------------------------------------- evaluation
SUBP = [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"),
        ("2023-24", "2023-01-01", "2025-01-01")]


def _sh(d):
    d = d.dropna()
    return float(d.mean() / d.std() * np.sqrt(365)) if len(d) > 20 and d.std() > 0 else np.nan


def evaluate(W, tf, name, family, U=None, R=None, full=True, extra=None):
    """Run base / 2x cost / lag-1 backtests plus decomposition; append one row to grid_log.csv."""
    res = lab.backtest(W, tf)
    m = res["metrics"]
    df = res["df"]
    d = lab.to_daily(df["r"], tf)
    row = {"family": family, "name": name, "tf": tf}
    for k in ["start", "end", "cagr", "vol", "sharpe", "maxdd", "t_nw", "skew", "kurt", "turnover_per_year",
              "cost_drag_per_year", "funding_per_year", "avg_gross", "avg_net", "avg_n_pos", "pct_pos_months"]:
        row[k] = m.get(k)
    bpy = lab.BARS_PER_YEAR[tf]
    row["gross_price_per_year"] = float(df["gross_r"].mean() * bpy)
    row["sharpe_gross_nocost"] = _sh(lab.to_daily(df["gross_r"] + df["fund_r"], tf))
    for lab_, a, b in SUBP:
        row["sh_" + lab_] = _sh(d[(d.index >= pd.Timestamp(a, tz="UTC")) & (d.index < pd.Timestamp(b, tz="UTC"))])
    yr = lab.yearly(df["r"], tf)
    row["yearly"] = "; ".join(f"{y}: {v:+.1%}" for y, v in yr["ret"].items())
    if full:
        row["sharpe_cost2x"] = lab.backtest(W, tf, cost=0.0014)["metrics"]["sharpe"]
        row["sharpe_lag1"] = lab.backtest(W, tf, extra_lag=1)["metrics"]["sharpe"]
        rl = lab.backtest(W.clip(lower=0), tf)
        rs = lab.backtest(W.clip(upper=0), tf)
        row["long_leg_sharpe"] = rl["metrics"].get("sharpe")
        row["short_leg_sharpe"] = rs["metrics"].get("sharpe")
        row["long_leg_ann"] = float(rl["df"]["r"].mean() * bpy)
        row["short_leg_ann"] = float(rs["df"]["r"].mean() * bpy)
        if U is not None and R is not None:
            ew = ew_return(R, U)
            ew.index = ew.index + pd.Timedelta({"1d": "1D", "4h": "4h", "1h": "1h"}[tf])
            Wf = W.reindex(index=R.index, columns=R.columns).fillna(0)
            gl = pd.Series(Wf.clip(lower=0).sum(axis=1).values, index=ew.index)
            gs = pd.Series((-Wf.clip(upper=0)).sum(axis=1).values, index=ew.index)
            lex = (rl["df"]["r"] - (gl * ew).reindex(rl["df"].index).fillna(0))
            sex = (rs["df"]["r"] + (gs * ew).reindex(rs["df"].index).fillna(0))
            row["long_leg_excess_vs_EW_sharpe"] = _sh(lab.to_daily(lex, tf))
            row["short_leg_excess_vs_EW_sharpe"] = _sh(lab.to_daily(sex, tf))
            row["long_leg_excess_ann"] = float(lex.mean() * bpy)
            row["short_leg_excess_ann"] = float(sex.mean() * bpy)
        c = controls()
        j = pd.concat([d.rename("s"), c], axis=1).dropna()
        row["corr_btc"] = float(j["s"].corr(j["btc"]))
        row["corr_xsmom"] = float(j["s"].corr(j["xsmom14"]))
        Wf = W.reindex(index=df.index - pd.Timedelta({"1d": "1D", "4h": "4h", "1h": "1h"}[tf])).abs()
        mn = Wf.where(Wf > 1e-9).min(axis=1).dropna()
        row["min_equity_usd"] = float(10 / mn.median()) if len(mn) else np.nan
        row["min_equity_usd_p90"] = float(10 / mn.quantile(0.1)) if len(mn) else np.nan
    if extra:
        row.update(extra)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    return row, res


def fmt(row):
    g = lambda k, f="{:.2f}": (f.format(row[k]) if row.get(k) is not None and row.get(k) == row.get(k) else "nan")
    return (f"{row['name']:42s} SR {g('sharpe')} (gross {g('sharpe_gross_nocost')}) 2x {g('sharpe_cost2x')} lag1 {g('sharpe_lag1')} | "
            f"20-21 {g('sh_2020-21')} 22 {g('sh_2022')} 23-24 {g('sh_2023-24')} | CAGR {g('cagr','{:+.1%}')} DD {g('maxdd','{:.0%}')} "
            f"cost {g('cost_drag_per_year','{:.1%}')} fund {g('funding_per_year','{:+.1%}')} | L {g('long_leg_excess_vs_EW_sharpe')} "
            f"S {g('short_leg_excess_vs_EW_sharpe')} | cBTC {g('corr_btc')} cMOM {g('corr_xsmom')} npos {g('avg_n_pos','{:.0f}')}")
