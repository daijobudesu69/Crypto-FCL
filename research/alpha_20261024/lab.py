"""Shared research harness for the 2026-10-24 alpha hunt.

    import sys; sys.path.insert(0, r"C:\\Crypto data\\backtest data and more\\alpha_20261024")
    import lab
    C = lab.load("close", "1d")            # wide panel, index = bar open ts (UTC), columns = symbols
    U = lab.universe("1d", top_k=50)       # point-in-time tradable universe (bool panel)
    W = ...                                # target weights decided at the CLOSE of bar t
    res = lab.backtest(W, "1d")            # costs + real funding, returns dict with daily returns + metrics
    lab.report(res)                        # one-line summary + yearly table

SPLIT GUARD: research period = 2020-01-01 .. 2024-12-31. Every loader truncates at
lab.IS_END unless the environment variable ALPHA_FORWARD=1 is set. Hunters must NEVER set it
and must never read data/ parquet files directly. The forward test (2025-01-01 ->) is run once,
by the lead, on the frozen spec.

Timing convention (no lookahead): every panel value at row t is known at the close of bar t
(= ts + tf). A weight W[t] is traded at that close and earns the bar t+1 return
close[t+1]/close[t]-1 and pays funding fund[t+1] (settlements inside (close t, close t+1]).
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
IS_END = pd.Timestamp("2025-01-01", tz="UTC")          # exclusive
FWD_START = IS_END
BARS_PER_YEAR = {"1d": 365, "4h": 365 * 6, "1h": 365 * 24, "hype_1d": 365}
TF_DELTA = {"1d": "1D", "4h": "4h", "1h": "1h", "hype_1d": "1D"}
COST = 0.0007                                           # per side: taker 0.05% + slippage 0.02%
FIELDS = ["open", "high", "low", "close", "qv", "tbqv", "trades", "fund", "oi_usd", "top_pos_ls",
          "top_acct_ls", "global_ls", "taker_ls_vol", "prem", "prem_mean", "spot_close", "spot_qv"]


def forward_enabled():
    return os.environ.get("ALPHA_FORWARD") == "1"


_mem = {}


def load(field, tf="1d", cols=None):
    """Wide float32 panel. Truncated at IS_END unless ALPHA_FORWARD=1."""
    key = (field, tf)
    if key not in _mem:
        df = pd.read_parquet(os.path.join(CACHE, tf, f"{field}.parquet"))
        if not forward_enabled():
            df = df[df.index < IS_END]
        _mem[key] = df
    df = _mem[key]
    return df[cols] if cols is not None else df


def clear():
    _mem.clear()


def returns(tf="1d"):
    c = load("close", tf)
    return c.pct_change(fill_method=None)


def universe(tf="1d", min_hist_days=60, top_k=None, min_qv_usd=None, lookback_days=30):
    """Point-in-time eligibility known at the close of bar t.

    eligible = listed for >= min_hist_days, has a close at t, and (optionally) its trailing
    lookback_days median daily quote volume ranks within top_k / exceeds min_qv_usd.
    NOTE: the symbol list itself is the 150 coins still alive in Sep 2026 (survivorship bias,
    accepted by the user). Point-in-time liquidity ranking reduces but cannot remove it.
    """
    bpd = {"1d": 1, "4h": 6, "1h": 24, "hype_1d": 1}[tf]
    c = load("close", tf)
    qv = load("qv", tf)
    hist = c.notna().cumsum()
    elig = c.notna() & (hist >= min_hist_days * bpd)
    if top_k is not None or min_qv_usd is not None:
        dqv = qv.rolling(lookback_days * bpd, min_periods=max(1, lookback_days * bpd // 2)).median() * bpd
        if min_qv_usd is not None:
            elig &= dqv >= min_qv_usd
        if top_k is not None:
            rk = dqv.where(elig).rank(axis=1, ascending=False)
            elig &= rk <= top_k
    return elig


def hype_symbols():
    """Binance symbols that also trade as a perp on Hyperliquid (HYPE), the execution venue."""
    m = pd.read_csv(os.path.join(HERE, "hype_map.csv"))
    return m.loc[m["hype"].notna(), "binance"].tolist()


def hype_universe(tf="1d", **kw):
    """universe() restricted to coins that are listed on HYPE today."""
    U = universe(tf, **kw)
    keep = U.columns.isin(hype_symbols())
    return U & pd.Series(keep, index=U.columns)


HYPE_COST = 0.00065                                     # per side: HYPE taker 0.045% + slippage 0.02%


def xs_rank(x, mask=None):
    """Cross-sectional percentile rank in [0,1] among valid (mask) names each bar."""
    if mask is not None:
        x = x.where(mask)
    return x.rank(axis=1, pct=True)


def xs_demean_weights(score, mask=None, gross=1.0):
    """Dollar-neutral weights proportional to the cross-sectionally demeaned score, |w| sums to gross."""
    s = score.where(mask) if mask is not None else score
    s = s.sub(s.mean(axis=1), axis=0)
    return s.div(s.abs().sum(axis=1), axis=0).fillna(0.0) * gross


def quantile_ls_weights(score, mask=None, q=0.2, gross=1.0, long_only=False, short_only=False):
    """Long top-q, short bottom-q, equal weight; each side gross/2 (or gross if one-sided)."""
    r = xs_rank(score, mask)
    lo = (r >= 1 - q).astype(float)
    sh = (r <= q).astype(float)
    nl = lo.sum(axis=1).replace(0, np.nan)
    ns = sh.sum(axis=1).replace(0, np.nan)
    if long_only:
        return lo.div(nl, axis=0).fillna(0.0) * gross
    if short_only:
        return -sh.div(ns, axis=0).fillna(0.0) * gross
    return (lo.div(nl, axis=0).fillna(0.0) - sh.div(ns, axis=0).fillna(0.0)) * gross / 2


def backtest(W, tf="1d", cost=COST, funding=True, extra_lag=0, rebalance_every=1, start=None, end=None):
    """Vectorised portfolio backtest.

    W            target weights (fraction of equity, + long / - short) decided at close of bar t.
    extra_lag    additional bars of delay before trading (robustness: signal decay).
    rebalance_every  only trade every k bars (weights held, with drift, in between).
    Returns dict: r (net bar returns), gross_r, cost_r, fund_r, turnover, gross_exposure, n_pos, metrics.
    Costs use drift-adjusted turnover: |W[t] - W[t-1]*(1+r_i)/(1+R_p)|.
    """
    C = load("close", tf)
    R = C.pct_change(fill_method=None)
    F = load("fund", tf)
    W = W.reindex(index=C.index, columns=C.columns).fillna(0.0)
    if extra_lag:
        W = W.shift(extra_lag).fillna(0.0)
    if rebalance_every > 1:
        keep = np.arange(len(W)) % rebalance_every == 0
        W = W.where(pd.Series(keep, index=W.index), np.nan).ffill().fillna(0.0)
    # a name without a next-bar return cannot be held -> force flat (delist / missing bar)
    Rn = R.shift(-1)
    Wv = W.where(Rn.notna() | (W == 0), 0.0).values
    Rn = Rn.fillna(0.0).values
    Fn = F.shift(-1).fillna(0.0).values
    gross_r = (Wv * Rn).sum(axis=1)
    fund_r = -(Wv * Fn).sum(axis=1) if funding else np.zeros(len(W))
    # drifted previous weights
    prev = np.vstack([np.zeros(Wv.shape[1]), Wv[:-1]])
    Rcur = np.vstack([np.zeros(Wv.shape[1]), Rn[:-1]])
    port_prev = np.concatenate([[0.0], gross_r[:-1]])
    drift = prev * (1 + Rcur) / (1 + port_prev)[:, None]
    turn = np.abs(Wv - drift).sum(axis=1)
    cost_r = turn * cost
    net = gross_r + fund_r - cost_r
    # net[t] is realised over bar t+1 -> shift index forward one bar
    ix = C.index
    out = pd.DataFrame({"r": net, "gross_r": gross_r, "fund_r": fund_r, "cost_r": cost_r,
                        "turnover": turn, "gross_exposure": np.abs(Wv).sum(axis=1),
                        "net_exposure": Wv.sum(axis=1), "n_pos": (Wv != 0).sum(axis=1)}, index=ix)
    out.index = out.index + pd.Timedelta(TF_DELTA[tf])
    out = out.iloc[:-1]
    if start is not None:
        out = out[out.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        out = out[out.index < pd.Timestamp(end, tz="UTC")]
    # trim leading flat period
    nz = np.flatnonzero(out["gross_exposure"].values > 0)
    if len(nz):
        out = out.iloc[nz[0]:]
    res = {"df": out, "tf": tf}
    res["metrics"] = metrics(out["r"], tf, out)
    return res


def to_daily(r, tf):
    if tf in ("1d", "hype_1d"):
        return r
    return (1 + r).groupby(r.index.floor("1D")).prod() - 1


def metrics(r, tf="1d", df=None):
    r = r.dropna()
    if len(r) < 10:
        return {}
    d = to_daily(r, tf)
    eq = (1 + d).cumprod()
    yrs = len(d) / 365
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = d.std() * np.sqrt(365)
    sh = d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else np.nan
    dn = d[d < 0].std() * np.sqrt(365)
    dd = (eq / eq.cummax() - 1).min()
    m = (1 + d).groupby(d.index.to_period("M")).prod() - 1 if d.index.tz is None else \
        (1 + d).groupby(d.index.tz_localize(None).to_period("M")).prod() - 1
    t_nw = newey_west_t(d.values, lags=10)
    out = {"start": str(d.index[0].date()), "end": str(d.index[-1].date()), "days": len(d),
           "cagr": cagr, "vol": vol, "sharpe": sh, "sortino": d.mean() * 365 / dn if dn > 0 else np.nan,
           "maxdd": dd, "calmar": cagr / abs(dd) if dd < 0 else np.nan, "t_nw": t_nw,
           "skew": float(pd.Series(d).skew()), "kurt": float(pd.Series(d).kurt()),
           "pct_pos_months": float((m > 0).mean()), "worst_month": float(m.min()), "best_month": float(m.max())}
    if df is not None:
        bpy = BARS_PER_YEAR[tf]
        out.update({"turnover_per_year": float(df["turnover"].mean() * bpy),
                    "cost_drag_per_year": float(df["cost_r"].mean() * bpy),
                    "funding_per_year": float(df["fund_r"].mean() * bpy),
                    "avg_gross": float(df["gross_exposure"].mean()),
                    "avg_net": float(df["net_exposure"].mean()),
                    "avg_n_pos": float(df["n_pos"].mean())})
    return out


def yearly(r, tf="1d"):
    d = to_daily(r.dropna(), tf)
    g = d.groupby(d.index.year)
    return pd.DataFrame({"ret": g.apply(lambda x: (1 + x).prod() - 1),
                         "sharpe": g.apply(lambda x: x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan),
                         "maxdd": g.apply(lambda x: ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min())})


def newey_west_t(x, lags=10):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < lags + 2:
        return np.nan
    mu = x.mean(); e = x - mu
    s = e @ e / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / n
    return mu / np.sqrt(s / n) if s > 0 else np.nan


def deflated_sharpe(sr_daily_ann, n_trials, T_days, skew=0.0, kurt=3.0, sr_var_trials=None):
    """Bailey & Lopez de Prado deflated Sharpe ratio (probability the true SR > expected max of
    n_trials null strategies). sr inputs annualised; kurt = raw (non-excess) kurtosis."""
    from scipy.stats import norm
    sr = sr_daily_ann / np.sqrt(365)
    if sr_var_trials is None:
        sr_var_trials = (0.5 / np.sqrt(365)) ** 2      # assume trials' annual SR dispersion ~0.5
    emc = 0.5772156649
    n = max(int(n_trials), 1)
    sr0 = np.sqrt(sr_var_trials) * ((1 - emc) * norm.ppf(1 - 1 / n) + emc * norm.ppf(1 - 1 / (n * np.e))) if n > 1 else 0.0
    z = (sr - sr0) * np.sqrt(T_days - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(norm.cdf(z))


def controls():
    """Daily control return series for correlation checks: BTC buy&hold and 14d XS momentum
    (top-50 universe, quintile long/short, daily rebalance, net of costs + funding)."""
    c = load("close", "1d")
    btc = c["BTCUSDT"].pct_change(fill_method=None)
    U = universe("1d", top_k=50)
    mom = c / c.shift(14) - 1
    r = backtest(quantile_ls_weights(mom, U, q=0.2), "1d")["df"]["r"]
    return pd.DataFrame({"btc": btc, "xsmom14": r}).dropna(how="all")


def report(res, name=""):
    m = res["metrics"]
    print(f"{name:40s} CAGR {m['cagr']:+.1%}  Sharpe {m['sharpe']:.2f}  t_NW {m['t_nw']:.2f}  MaxDD {m['maxdd']:.1%}  "
          f"turn/yr {m.get('turnover_per_year', np.nan):.0f}  cost/yr {m.get('cost_drag_per_year', np.nan):.1%}  "
          f"fund/yr {m.get('funding_per_year', np.nan):+.1%}  gross {m.get('avg_gross', np.nan):.2f}  "
          f"npos {m.get('avg_n_pos', np.nan):.1f}")
    print(yearly(res["df"]["r"], res["tf"]).round(3).T.to_string())
