"""FAMILY A - LISTING AGE / SUPPLY OVERHANG (round 2, HYPE candles only).

Hypothesis: newly listed tokens (low float, high FDV, scheduled VC/team unlocks, airdrop farmers)
face persistent supply overhang in their first months -> young coins underperform mature coins.
Other side: retail/attention buyers of new listings; arbitrage limited because shorting new perps is
expensive (high, volatile funding, squeeze risk). Signal = age only (static metadata derived from the
date of the first candle), so it uses no forbidden field.

Universe: lab.hype_universe("1d", min_hist_days=30, top_k=50) (robustness top_k 30/75 on the pick).
Cost lab.HYPE_COST + Binance funding as cost. Age definitions in ages.py (raw / clean).

PRE-REGISTERED GRID (written before running; 38 configs):
 A1 continuous: W = xs_demean_weights(log(min(age, cap)))  (long old / short young, $-neutral, gross 1)
    cap in {365, 730} x age in {raw, clean} x rebalance in {1, 7}                                = 8
 A2 bucket: short EW coins with lo <= age < hi, long EW "mature" coins, each leg gross 0.5, flat
    if either leg empty, rebalance every 7d.
    lo in {30, 60} x hi in {180, 270, 365} x age in {raw, clean} x long side in
    {MAT = age >= 730, REST = age >= hi}                                                          = 24
 A3 beta-neutral A2 (lo=30, hi=365, MAT): each leg scaled to beta 0.5 vs BTC (trailing 90d beta,
    leg gross cap 1.0), age in {raw, clean}                                                       = 2
 A4 short young (30<=age<365) EW gross 0.5 vs long BTC 0.5, age in {raw, clean}                    = 2
 A5 Fama-MacBeth diagnostic: daily XS regression of next-day return on ranks of
    [log age capped 730, mom14, vol30, log qv30] -> NW t of age slope, age in {raw, clean}        = 2
Selection rule for the family "best": highest full-period Sharpe among A1-A4; it is then stressed
(2x cost, lag 1, top_k 30/75, rebalance 1, 16-position version 8 long / 8 short).
"""
import numpy as np, pandas as pd, json, warnings
import r2lib, ages
warnings.filterwarnings("ignore")
lab = r2lib.lab
N_TRIALS = 38

C = lab.load("close", "1d")
R = C.pct_change(fill_method=None)
QV = lab.load("qv", "1d")
AGE = {k: ages.age_panel(k) for k in ("raw", "clean")}


def univ(top_k=50):
    return lab.hype_universe("1d", min_hist_days=30, top_k=top_k)


def bucket_weights(A, U, lo, hi, long_side="MAT", gross=1.0):
    young = U & (A >= lo) & (A < hi)
    if long_side == "MAT":
        old = U & (A >= 730)
    elif long_side == "REST":
        old = U & (A >= hi)
    elif long_side == "BTC":
        old = pd.DataFrame(False, index=U.index, columns=U.columns)
        old[r2lib.BTC] = U[r2lib.BTC]
    ny = young.sum(axis=1); no = old.sum(axis=1)
    ok = (ny > 0) & (no > 0)
    Wl = old.astype(float).div(no.replace(0, np.nan), axis=0)
    Ws = young.astype(float).div(ny.replace(0, np.nan), axis=0)
    W = (Wl - Ws).mul(ok.astype(float), axis=0).fillna(0.0) * gross / 2
    return W


def rolling_beta(win=90):
    m = R[r2lib.BTC]
    Rm = R
    M = pd.DataFrame(np.repeat(m.values[:, None], R.shape[1], 1), index=R.index, columns=R.columns).where(Rm.notna())
    Rm = Rm.where(M.notna())
    mp = int(win * 2 / 3)
    ex = Rm.rolling(win, min_periods=mp).mean(); em = M.rolling(win, min_periods=mp).mean()
    cov = (Rm * M).rolling(win, min_periods=mp).mean() - ex * em
    vm = (M * M).rolling(win, min_periods=mp).mean() - em * em
    return cov / vm


def beta_neutral(W, beta, leg_beta=0.5, cap=1.0):
    Wl = W.clip(lower=0); Ws = -W.clip(upper=0)
    Wl = Wl.div(Wl.sum(1).replace(0, np.nan), axis=0); Ws = Ws.div(Ws.sum(1).replace(0, np.nan), axis=0)
    b = beta.fillna(beta.median(axis=1).mean())
    bl = (Wl * b).sum(1); bs = (Ws * b).sum(1)
    kl = (leg_beta / bl.where(bl > 0.1)).clip(upper=cap); ks = (leg_beta / bs.where(bs > 0.1)).clip(upper=cap)
    return (Wl.mul(kl, axis=0) - Ws.mul(ks, axis=0)).fillna(0.0)


def fama_macbeth(A, U):
    Rn = R.shift(-1)
    mom = C / C.shift(14) - 1
    vol = R.rolling(30, min_periods=20).std()
    size = np.log(QV.rolling(30, min_periods=15).median())
    feats = [np.log(A.clip(upper=730)), mom, vol, size]
    ranks = [f.where(U).rank(axis=1, pct=True) for f in feats]
    coefs = []
    for t in range(len(C) - 1):
        y = Rn.iloc[t].values
        X = np.column_stack([r.iloc[t].values for r in ranks])
        m = np.isfinite(y) & np.isfinite(X).all(1)
        if m.sum() < 12:
            continue
        Xm = X[m] - X[m].mean(0)
        if np.linalg.matrix_rank(Xm) < Xm.shape[1]:
            continue
        b = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), Xm]), y[m], rcond=None)[0]
        coefs.append(b[1:])
    coefs = np.array(coefs)
    return {nm: (float(coefs[:, i].mean() * 365), float(lab.newey_west_t(coefs[:, i], 10)))
            for i, nm in enumerate(["age", "mom14", "vol30", "size"])}, len(coefs)


def run_grid():
    U = univ(50)
    rows = []
    beta = rolling_beta(90)

    def go(cfg, W, rb):
        o, _ = r2lib.evaluate(W, cfg, U=U, rebalance_every=rb, full=False)
        o.update({"family": "A"})
        rows.append(o); r2lib.log(o)
        print(r2lib.fmt(o, keys=("sharpe", "cagr", "maxdd", "sh_2020-21", "sh_2022", "sh_2023-24",
                                  "avg_n_pos", "turnover_per_year", "funding_per_year")), flush=True)

    for kind in ("raw", "clean"):
        A = AGE[kind]
        for cap in (365, 730):
            S = np.log(A.clip(lower=1, upper=cap))
            W = lab.xs_demean_weights(S, U & A.notna())
            for rb in (1, 7):
                go(f"A1_{kind}_cap{cap}_rb{rb}", W, rb)
    for kind in ("raw", "clean"):
        A = AGE[kind]
        for lo in (30, 60):
            for hi in (180, 270, 365):
                for ls in ("MAT", "REST"):
                    go(f"A2_{kind}_{lo}-{hi}_{ls}", bucket_weights(A, U, lo, hi, ls), 7)
    for kind in ("raw", "clean"):
        A = AGE[kind]
        W = beta_neutral(bucket_weights(A, U, 30, 365, "MAT"), beta)
        go(f"A3_{kind}_30-365_MAT_betaneutral", W, 7)
    for kind in ("raw", "clean"):
        A = AGE[kind]
        go(f"A4_{kind}_30-365_vsBTC", bucket_weights(A, U, 30, 365, "BTC"), 7)
    fm = {}
    for kind in ("raw", "clean"):
        res, n = fama_macbeth(AGE[kind], U)
        fm[kind] = res
        print(f"A5_FM_{kind} (n days {n}): " + "  ".join(f"{k}: slope/yr {v[0]:+.3f} t {v[1]:+.2f}" for k, v in res.items()))
    pd.DataFrame(rows).to_csv(r2lib.os.path.join(r2lib.HERE, "famA_grid.csv"), index=False)
    json.dump(fm, open(r2lib.os.path.join(r2lib.HERE, "famA_fm.json"), "w"), indent=1)


if __name__ == "__main__":
    run_grid()
