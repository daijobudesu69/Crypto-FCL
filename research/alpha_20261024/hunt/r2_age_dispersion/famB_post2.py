"""FAMILY B follow-up 2: survivorship sanity, decay, beta-neutral, small-account versions.

POST-HOC configs (counted, family total 48 + 5 = 53):
 old-majors spread: alts = the 14 pre-2020-03 legacy alts on HYPE (ETH BCH XRP LTC TRX ETC LINK XLM ADA
   XMR DASH ZEC ATOM BNB, survival to 2026 nearly certain ex-ante), signal from U50 / from majors only = 2
 B1_disp7_TERC with weekly rebalance                                                                   = 1
 beta-neutral spread (BTC leg sized by the alt basket's trailing 90d beta to BTC), TERC                 = 1
 11-position version (10 most liquid HYPE alts + BTC; $10 min at $200 equity), TERC                     = 1
Diagnostics (not configs): signal decay extra_lag 2/3/5/10, per-coin contribution of the alt leg,
 forward 7d spread return by state, signal-on-old-majors share.
"""
import numpy as np, pandas as pd, json, warnings
import r2lib
from famB_regime import C, R, U50, BTC, trail_pct, state, spread_w, ew_weights
from bhelp import disp, st_from, align
warnings.filterwarnings("ignore")
lab = r2lib.lab
N_TRIALS = 53
K = ("sharpe", "cagr", "maxdd", "sh_2020-21", "sh_2022", "sh_2023-24", "sharpe_cost2x", "sharpe_lag1", "turnover_per_year", "avg_n_pos")
out = {}
st7t = st_from(disp(U50, 7), kind="TERC")
st7m = st_from(disp(U50, 7), kind="MED")
first = C.apply(lambda s: s.first_valid_index())
majors = [s for s in first.index[first < pd.Timestamp("2020-03-01", tz="UTC")] if s in lab.hype_symbols() and s != BTC]
print("old majors:", majors)
Umaj = (C.notna() & (C.notna().cumsum() >= 60)) & pd.Series(C.columns.isin(majors + [BTC]), index=C.columns)

o, _ = r2lib.evaluate(spread_w(st7t, Umaj), "B1_disp7_TERC_oldmajors_sigU50", U=Umaj, full=True)
r2lib.log(o); print(r2lib.fmt(o, K)); print("   yearly", o["yearly"]); out["oldmaj_sigU50"] = o
o, _ = r2lib.evaluate(spread_w(st_from(disp(Umaj, 7), kind="TERC"), Umaj), "B1_disp7_TERC_oldmajors_sigMajors", U=Umaj, full=True)
r2lib.log(o); print(r2lib.fmt(o, K)); print("   yearly", o["yearly"]); out["oldmaj_sigMaj"] = o

W = spread_w(st7t, U50)
o, _ = r2lib.evaluate(W, "B1_disp7_TERC_rb7", U=U50, rebalance_every=7, full=True)
r2lib.log(o); print(r2lib.fmt(o, K)); out["rb7"] = o

# beta-neutral: BTC leg = beta_alt * 0.5 (capped 1.0)
alts = U50.copy(); alts[BTC] = False
Wa = ew_weights(alts)
ralt = (Wa.shift(1) * R).sum(axis=1)
rb = R[BTC]
beta = (ralt.rolling(90, min_periods=60).cov(rb) / rb.rolling(90, min_periods=60).var()).clip(0.5, 2.0)
s = st7t.fillna(0.0)
Wbn = Wa.mul(0.5 * s, axis=0)
Wbn[BTC] = Wbn[BTC] - (0.5 * s * beta.fillna(1.0)).clip(-1, 1)
o, _ = r2lib.evaluate(Wbn, "B1_disp7_TERC_betaneutral", U=U50, full=True)
r2lib.log(o); print(r2lib.fmt(o, K)); out["betaneutral"] = o
print(f"   mean trailing alt beta {beta.mean():.2f}")

U11 = lab.hype_universe("1d", top_k=11)
for nm, st in (("B1_disp7_TERC_11pos", st7t), ("B1_disp7_MED_11pos", st7m)):
    o, _ = r2lib.evaluate(spread_w(st, U11), nm, U=U11, full=True, n_trials=N_TRIALS)
    if "TERC" in nm:
        r2lib.log(o)
    print(r2lib.fmt(o, K), f"min_eq {o['min_equity_median']:.0f}"); print("   yearly", o["yearly"]); out[nm] = o

# decay
for L in (2, 3, 5, 10):
    r = lab.backtest(W, "1d", cost=r2lib.COST, extra_lag=L)["metrics"]
    print(f"   TERC extra_lag {L}: Sharpe {r['sharpe']:.2f}"); out[f"lag{L}"] = r["sharpe"]
Wm = spread_w(st7m, U50)
for L in (2, 3, 5, 10):
    r = lab.backtest(Wm, "1d", cost=r2lib.COST, extra_lag=L)["metrics"]
    print(f"   MED  extra_lag {L}: Sharpe {r['sharpe']:.2f}"); out[f"med_lag{L}"] = r["sharpe"]

# forward 7d spread return (alt EW - BTC) by MED state - overlapping, descriptive only
spr1 = ralt - rb
fwd7 = sum(spr1.shift(-k) for k in range(1, 8))
for sv in (1, -1):
    x = fwd7[st7m == sv]
    print(f"   MED state {sv:+d}: mean fwd 7d spread {x.mean():+.2%} (n {x.notna().sum()})")

# per-coin contribution of the alt leg (TERC)
Walt = W.copy(); Walt[BTC] = 0.0
contrib = (Walt * R.shift(-1).fillna(0.0)).sum().sort_values()
print("   alt-leg contribution top 8:", contrib.tail(8).round(3).to_dict())
print("   alt-leg contribution bottom 5:", contrib.head(5).round(3).to_dict())
print(f"   total {contrib.sum():.3f}, ex top-5 {contrib.iloc[:-5].sum():.3f}")
out["altleg_top8"] = contrib.tail(8).round(3).to_dict(); out["altleg_total"] = float(contrib.sum())
out["altleg_ex_top5"] = float(contrib.iloc[:-5].sum())
json.dump(out, open(r2lib.os.path.join(r2lib.HERE, "famB_post2.json"), "w"), indent=1, default=str)
