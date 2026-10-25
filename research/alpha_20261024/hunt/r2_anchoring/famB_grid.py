"""FAMILY B - RESIDUAL & INFORMATION-DISCRETENESS MOMENTUM, OHLCV only.

B(i) Idiosyncratic (residual) momentum (Blitz-Huij-Martens 2011; Gutierrez-Prinsky 2007).
  Rationale: plain XS momentum in crypto is dominated by BTC-beta timing (high-beta alts win in
  up-markets, lose in crashes -> momentum crashes). Coin-specific news (listings, unlocks,
  narratives) diffuses slowly through a fragmented, attention-limited retail market; the
  residual vs BTC isolates that under-reaction and should be less crash-prone. Counterparty:
  investors who trade alts as a BTC-beta bucket and ignore coin-specific drift.
  eps_t = r_i,t - beta_{i,t-1} * r_BTC,t, beta = trailing 90d OLS beta vs BTC (min 60 obs).
  PRE-REGISTERED GRID (24): L in {30,60,90} formation days x skip in {3,7} x
     score in {raw = sum eps, scaled = sum eps / std eps (Blitz)} x rebal in {1,7}.
  fixed: lab.hype_universe top_k=50 (BTC excluded from ranking), quintile q=0.2, then
         beta-neutral leg scaling (long_gross*beta_L = short_gross*beta_S, gross 1, legs clipped
         to 25-75%), HYPE cost + funding.

B(ii) Frog-in-the-pan / information discreteness (Da, Gurun, Warachka 2014).
  Rationale: limited attention - a stream of small same-sign moves is under-noticed, big jumps
  grab attention and get priced. ID = sign(PRET) * (%neg days - %pos days); low ID = continuous.
  PRE-REGISTERED GRID (18): variant x L in {30,60,90} x rebal in {1,7}; skip fixed = 3 days.
    SGN : score = %pos - %neg days over the window (sign / consistency momentum), quintile L/S.
    FIP : Da et al. double sort: winners = top tercile PRET, losers = bottom tercile; long the
          continuous half (low ID) of winners, short the continuous half of losers.
    IDS : momentum-neutral ID spread: long continuous winners + discrete losers, short discrete
          winners + continuous losers (pure test of whether ID adds to momentum).
  fixed: hype top-50, equal weight, $-neutral, gross 1, HYPE cost + funding.
Total family B = 42 configs.
"""
import itertools
import numpy as np, pandas as pd
import common as cm
import lab

c = lab.load("close", "1d")
R = cm.R()
u50 = cm.U(50)
beta = cm.rolling_beta(90, 60)
rb = R["BTCUSDT"]
eps = R - beta.shift(1).mul(rb, axis=0)
eps["BTCUSDT"] = np.nan
u_nobtc = u50.copy()
u_nobtc["BTCUSDT"] = False


def resid_score(L, skip, kind):
    minp = int(0.8 * L)
    s = eps.rolling(L, min_periods=minp).sum()
    if kind == "scaled":
        s = s / eps.rolling(L, min_periods=minp).std()
    return s.shift(skip)


def fip_parts(L, skip):
    minp = int(0.8 * L)
    rr = R.shift(skip)
    n = rr.notna().astype(float).rolling(L, min_periods=1).sum()
    pos = (rr > 0).astype(float).rolling(L, min_periods=1).sum() / n
    neg = (rr < 0).astype(float).rolling(L, min_periods=1).sum() / n
    pret = c.shift(skip) / c.shift(skip + L) - 1
    ok = (n >= minp) & pret.notna()
    idd = np.sign(pret) * (neg - pos)
    return pos.where(ok), neg.where(ok), pret.where(ok), idd.where(ok)


def group_weights(groups_long, groups_short):
    """equal weight inside each side; side gross 0.5."""
    lo = sum(g.astype(float) for g in groups_long)
    sh = sum(g.astype(float) for g in groups_short)
    nl = lo.sum(axis=1).replace(0, np.nan)
    ns = sh.sum(axis=1).replace(0, np.nan)
    return (lo.div(nl, axis=0) - sh.div(ns, axis=0)).fillna(0.0) * 0.5


def fip_weights(L, skip, variant, u=None):
    u = u50 if u is None else u
    pos, neg, pret, idd = fip_parts(L, skip)
    if variant == "SGN":
        return lab.quantile_ls_weights(pos - neg, u, q=0.2)
    pr = lab.xs_rank(pret, u & pret.notna())
    win = pr >= 2 / 3
    los = pr <= 1 / 3
    idw = idd.where(win).rank(axis=1, pct=True)
    idl = idd.where(los).rank(axis=1, pct=True)
    cont_w, disc_w = idw <= 0.5, idw > 0.5
    cont_l, disc_l = idl <= 0.5, idl > 0.5
    if variant == "FIP":
        return group_weights([cont_w], [cont_l])
    if variant == "IDS":
        return group_weights([cont_w, disc_l], [disc_w, cont_l])
    raise ValueError(variant)


def resid_weights(L, skip, kind, u=None, q=0.2):
    u = u_nobtc if u is None else (u & ~pd.Series(u.columns == "BTCUSDT", index=u.columns))
    W = lab.quantile_ls_weights(resid_score(L, skip, kind), u, q=q)
    return cm.beta_neutral(W, beta)


if __name__ == "__main__":
    rows = []
    for L, skip, kind, rebal in itertools.product([30, 60, 90], [3, 7], ["raw", "scaled"], [1, 7]):
        W = resid_weights(L, skip, kind)
        o, _ = cm.quick(W, rebal)
        o.update({"sub": "RESID", "L": L, "skip": skip, "kind": kind, "rebal": rebal})
        rows.append(o)
        print(f"RESID L={L:2d} sk={skip} {kind:6s} rb={rebal}  SR {o['sharpe']:+.2f}  20-21 {o['sh_2020-21']:+.2f}  "
              f"22 {o['sh_2022']:+.2f}  23-24 {o['sh_2023-24']:+.2f}  turn {o['turn']:.0f}  fund {o['fund_yr']:+.1%}",
              flush=True)
    for variant, L, rebal in itertools.product(["SGN", "FIP", "IDS"], [30, 60, 90], [1, 7]):
        W = fip_weights(L, 3, variant)
        o, _ = cm.quick(W, rebal)
        o.update({"sub": variant, "L": L, "skip": 3, "kind": "-", "rebal": rebal})
        rows.append(o)
        print(f"{variant:5s} L={L:2d} sk=3 rb={rebal}  SR {o['sharpe']:+.2f}  20-21 {o['sh_2020-21']:+.2f}  "
              f"22 {o['sh_2022']:+.2f}  23-24 {o['sh_2023-24']:+.2f}  turn {o['turn']:.0f}  npos {o['npos']:.1f}",
              flush=True)
    g = pd.DataFrame(rows)
    g.to_csv("famB_grid.csv", index=False)
    print(len(g), "configs")
