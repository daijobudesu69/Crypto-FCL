"""v1_reimpl_audit — independent re-implementation of "Funding-crowding L/S" (hunter h1).

Written from the rules text only (hunter scripts NOT read before this file was written).
Uses only lab.load panels (close, qv, fund) and an own portfolio / cost / funding loop.
No lab.backtest, no lab.universe, no lab.quantile_ls_weights in this file.

Rules (as given):
  daily at 00:00 UTC; point-in-time top-50 Binance USDT perps by trailing 30d median quote volume
  (>= 60 days listed); F14 = sum of last 14 days of funding settlements known at the close;
  short top third by F14 (eq weight, -0.5 gross), long bottom third (eq weight, +0.5 gross);
  rebalance daily; 0.07%/side + actual funding.

Own conventions (decided before seeing any results):
  * fund[t] (cache) = settlements in (t, t+1d]; known at close t (the last one is AT the close).
  * Position chosen at close t is held over bar t+1: earns close[t+1]/close[t]-1, pays fund[t+1].
  * Universe: close at t present, >= 60 non-missing daily closes up to and incl. t,
    30-row trailing median of qv (min 30 obs; variant min 15), rank top 50 among eligible.
  * Terciles: k = floor(n/3) names per side where n = #universe names with a valid F14.
    Sort by F14; ties broken by a deterministic secondary key (variants tested).
  * Account loop in dollar notionals: equity E, positions drift with own price; trade cost on
    |target - drifted| * cost; funding on held notional; missing next close -> position is
    HELD (marked flat for the day, price gap realised when the next close appears) instead of
    being force-closed (variant: force-close like lab).
"""
import sys, os, json
import numpy as np, pandas as pd

sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab

OUT = os.path.dirname(os.path.abspath(__file__))


def build_universe(C, Q, top_k=50, min_hist=60, lb=30, minp=30):
    hist = C.notna().cumsum()
    elig = C.notna() & (hist >= min_hist)
    med = Q.astype("float64").rolling(lb, min_periods=minp).median()
    med = med.where(elig)
    # rank descending, ties -> first by column order (rare for float qv)
    rk = med.rank(axis=1, ascending=False, method="first")
    return elig & (rk <= top_k) & med.notna()


def f14_signal(F, n=14, exact=True):
    f = F.astype("float64")
    if exact:
        # integer arithmetic in units of 1e-8 to make ties exact (avoid float running-sum noise)
        fi = np.round(f * 1e8)
        s = fi.rolling(n, min_periods=n).sum() / 1e8
    else:
        s = F.rolling(n, min_periods=n).sum()
    return s


def target_weights(S, U, tie="symbol", gross=1.0, seed=0, frac=1/3):
    """Short top-k by S, long bottom-k by S; k=floor(n*frac); returns dict t -> Series."""
    rng = np.random.default_rng(seed)
    cols = S.columns
    W = np.zeros(S.shape)
    Sv = S.values; Uv = U.values
    for i in range(len(S)):
        m = Uv[i] & ~np.isnan(Sv[i])
        idx = np.flatnonzero(m)
        n = len(idx)
        k = int(np.floor(n * frac))
        if k < 1:
            continue
        s = Sv[i, idx]
        if tie == "symbol":
            sec = np.arange(n)            # column order
        elif tie == "rsymbol":
            sec = -np.arange(n)
        elif tie == "random":
            sec = rng.permutation(n)
        order = np.lexsort((sec, s))      # ascending by s, then secondary
        lo = idx[order[:k]]               # lowest F14 -> long
        hi = idx[order[-k:]]              # highest F14 -> short
        W[i, lo] = +0.5 * gross / k
        W[i, hi] = -0.5 * gross / k
    return pd.DataFrame(W, index=S.index, columns=cols)


def avg_rank_weights(S, U, gross=1.0, q=1/3):
    """lab-like rule: pct rank (average ties), long r<=q... (score=-F14 -> long top) """
    r = (-S).where(U).rank(axis=1, pct=True)
    lo = (r >= 1 - q).astype(float); sh = (r <= q).astype(float)
    nl = lo.sum(axis=1).replace(0, np.nan); ns = sh.sum(axis=1).replace(0, np.nan)
    return (lo.div(nl, axis=0).fillna(0) - sh.div(ns, axis=0).fillna(0)) * gross / 2


def run_account(W, C, F, cost=0.0007, force_close_missing=False, funding=True, lag=0):
    """Own dollar-notional loop. W row t = target chosen at close t. Returns daily df indexed by
    the close at which the day's P&L is realised (t+1)."""
    if lag:
        W = W.shift(lag).fillna(0.0)
    Cv = C.values.astype("float64")
    Fv = F.values.astype("float64")
    Wv = W.reindex(index=C.index, columns=C.columns).fillna(0.0).values
    T, N = Cv.shape
    last_px = np.full(N, np.nan)          # last known close per name
    E = 1.0
    pos = np.zeros(N)                     # $ notional after drift (marked at last_px)
    rows = []
    for t in range(T - 1):
        px = Cv[t]
        has = ~np.isnan(px)
        last_px[has] = px[has]
        tgt = Wv[t] * E
        # cannot open/hold a name without a price at t (keep existing if no price: frozen)
        no_px = ~has
        tgt[no_px & (pos == 0)] = 0.0
        tgt[no_px & (pos != 0)] = pos[no_px & (pos != 0)]   # frozen: cannot trade it
        if force_close_missing:
            nxt_missing = np.isnan(Cv[t + 1])
            tgt[nxt_missing & has] = 0.0
        trade = np.abs(tgt - pos)
        c = trade.sum() * cost
        pos = tgt.copy()
        # P&L over bar t+1
        nxt = Cv[t + 1]
        ok = ~np.isnan(nxt) & ~np.isnan(last_px)
        ret = np.zeros(N)
        ret[ok] = nxt[ok] / last_px[ok] - 1.0      # gap realised when price reappears
        pnl = (pos * ret).sum()
        fnd = -(pos * np.nan_to_num(Fv[t + 1])).sum() if funding else 0.0
        grossexp = np.abs(pos).sum() / E
        E_new = E + pnl + fnd - c
        rows.append((C.index[t + 1], (E_new / E) - 1.0, pnl / E, fnd / E, -c / E, trade.sum() / E,
                     grossexp, pos.sum() / E, int((pos != 0).sum()),
                     (pos[pos > 0] * ret[pos > 0]).sum() / E, (pos[pos < 0] * ret[pos < 0]).sum() / E,
                     -(pos[pos > 0] * np.nan_to_num(Fv[t + 1][pos > 0])).sum() / E,
                     -(pos[pos < 0] * np.nan_to_num(Fv[t + 1][pos < 0])).sum() / E))
        pos = pos * (1.0 + ret)
        pos[ok] = pos[ok]  # drifted
        E = E_new
    df = pd.DataFrame(rows, columns=["ts", "r", "price_r", "fund_r", "cost_r", "turnover", "gross", "net",
                                     "n_pos", "long_px", "short_px", "long_fund", "short_fund"]).set_index("ts")
    # label = open of the earning bar t+1 (same convention as lab.backtest)
    nz = np.flatnonzero(df["gross"].values > 0)
    return df.iloc[nz[0]:] if len(nz) else df


def stats(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    out = {"start": str(r.index[0].date()), "end": str(r.index[-1].date()), "days": len(r),
           "sharpe": float(r.mean() / r.std() * np.sqrt(365)),
           "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
           "maxdd": float((eq / eq.cummax() - 1).min()),
           "vol": float(r.std() * np.sqrt(365)),
           "skew": float(r.skew()), "kurt_ex": float(r.kurt())}
    g = r.groupby(r.index.year)
    out["yearly_ret"] = {int(y): float((1 + x).prod() - 1) for y, x in g}
    out["yearly_sharpe"] = {int(y): float(x.mean() / x.std() * np.sqrt(365)) for y, x in g}
    return out


def fmt(name, s):
    yr = " ".join(f"{y}:{v:+.1%}" for y, v in s["yearly_ret"].items())
    return (f"{name:46s} SR {s['sharpe']:.2f} CAGR {s['cagr']:+.1%} DD {s['maxdd']:.1%} "
            f"[{s['start']}..] | {yr}")


if __name__ == "__main__":
    C = lab.load("close", "1d"); Q = lab.load("qv", "1d"); F = lab.load("fund", "1d")
    res = {}
    U = build_universe(C, Q)
    S = f14_signal(F, exact=True)
    print("universe size (median, min after 2020-03):", int(U.sum(axis=1)[U.index >= "2020-03-01"].median()),
          int(U.sum(axis=1)[U.index >= "2020-03-01"].min()))
    # exact ties statistics
    Sm = S.where(U)
    tie_frac = []
    for i in range(len(Sm)):
        v = Sm.iloc[i].dropna().values
        if len(v) >= 10:
            vals, cnt = np.unique(v, return_counts=True)
            tie_frac.append(cnt.max() / len(v))
    print("median share of universe in the largest exact F14 tie group:", np.median(tie_frac), "mean", np.mean(tie_frac))
    res["tie_share_median"] = float(np.median(tie_frac)); res["tie_share_mean"] = float(np.mean(tie_frac))

    runs = {}
    W0 = target_weights(S, U, tie="symbol")
    runs["base (own loop, hold thru gaps, tie=symbol)"] = run_account(W0, C, F)
    runs["force-close next-missing (lab-like)"] = run_account(W0, C, F, force_close_missing=True)
    runs["2x cost 0.14%"] = run_account(W0, C, F, cost=0.0014)
    runs["lag 1"] = run_account(W0, C, F, lag=1)
    runs["no funding"] = run_account(W0, C, F, funding=False)
    runs["tie=reverse symbol"] = run_account(target_weights(S, U, tie="rsymbol"), C, F)
    for sd in range(5):
        runs[f"tie=random seed{sd}"] = run_account(target_weights(S, U, tie="random", seed=sd), C, F)
    runs["float32 running-sum F14 (no exact ties)"] = run_account(target_weights(f14_signal(F, exact=False), U), C, F)
    runs["avg-rank pct terciles (lab rule), exact F14"] = run_account(avg_rank_weights(S, U), C, F)
    runs["avg-rank pct terciles, float32 F14"] = run_account(avg_rank_weights(f14_signal(F, exact=False), U), C, F)
    U15 = build_universe(C, Q, minp=15)
    runs["universe min_periods=15"] = run_account(target_weights(S, U15), C, F)
    # leg split from base
    b = runs["base (own loop, hold thru gaps, tie=symbol)"]
    long_r = b["long_px"] + b["long_fund"]; short_r = b["short_px"] + b["short_fund"]
    for name, df in runs.items():
        s = stats(df["r"]); res[name] = s
        print(fmt(name, s))
    print("IS window 2020-03-01.. (claimed start):")
    for name in ["base (own loop, hold thru gaps, tie=symbol)", "avg-rank pct terciles, float32 F14"]:
        df = runs[name]; s = stats(df["r"][df.index >= "2020-03-01"]); print(fmt(name + " [from 2020-03]", s))
    res["leg_long"] = stats(long_r); res["leg_short"] = stats(short_r)
    print(fmt("long leg (price+fund, gross 0.5)", res["leg_long"]))
    print(fmt("short leg (price+fund, gross 0.5)", res["leg_short"]))
    print("base decomposition per yr: price %.3f fund %.3f cost %.3f turnover/yr %.1f" % (
        b["price_r"].mean() * 365, b["fund_r"].mean() * 365, b["cost_r"].mean() * 365, b["turnover"].mean() * 365))
    print("short-leg funding per yr %.3f, long-leg funding per yr %.3f" % (b["short_fund"].mean() * 365, b["long_fund"].mean() * 365))
    print("avg n_pos", b["n_pos"].mean())
    pd.DataFrame({k: v["r"] for k, v in runs.items()}).to_csv(os.path.join(OUT, "reimpl_daily.csv"))
    W0.to_parquet(os.path.join(OUT, "W_reimpl_base.parquet"))
    with open(os.path.join(OUT, "reimpl_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
