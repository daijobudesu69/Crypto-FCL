"""FAMILY A - BASIS / PREMIUM (h3_basis_flow).  Research data 2020-2024 only via lab.py.

HYPOTHESIS: the perp premium index (impact-price perp vs spot index; the input to Binance funding)
measures leveraged-long demand. Rich premium = crowded levered longs paying up -> subsequent
underperformance (and longs bleed funding); discount = hedging / short pressure -> rebound.
Other side: retail levered longs paying for convexity/attention; arbs are capital-constrained so
the premium can persist. Funding is CLAMPED (0.01%/8h whenever |premium-0.01%|<0.05%), so prem_mean
carries information that funding does not inside the band -> check if it adds beyond funding.

PRE-REGISTERED GRID (written before running; universe = lab.universe('1d', min_hist_days=60,
top_k=50); weights = lab.quantile_ls_weights(q=0.2) $-neutral; daily rebalance unless noted;
contrarian sign: score = -signal):
 A1 pm_lvl_L      score=-rolling_mean(prem_mean, L)                      L in {1,3,7,14}      4
 A2 pm_z_L        score=-(pm_L - mean90(pm))/std90(pm), trailing         L in {1,3,7}         3
 A3 basis_lvl_L   basis=close/spot_close-1 ; score=-rolling_mean(basis,L) L in {1,3,7}         3
 A4 basis_z_L     trailing 90d z of basis_L                              L in {3,7}           2
 A5 pm_xfund_L    score=-(xsrank(pm_L) - xsrank(fund_L)) (premium beyond funding) L in {3,7}   2
 A6 demean        xs_demean_weights on A1 L=7 and A2 L=7                                      2
 A7 rebal3        rebalance_every=3 on A1 L=7 and A2 L=7                                      2
 A0 control       trailing funding factor: score=-sum(fund,7)                                 1
 A-TS  time series on BTC, ETH, EW top-50 basket: z = trailing-90d z of pm_L, L in {3,7};
       rule a: pos=-clip(z/2,-1,1) (long/short) ; rule b: pos=clip(-z/2,0,1) (long-only)     12
 TOTAL family A = 31 configs.  Diagnostics (not trading configs): rank IC at h=1,3,7d.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import h3lib as H
import lab

U = lab.universe("1d", min_hist_days=60, top_k=50)
C = lab.load("close", "1d")
PM = lab.load("prem_mean", "1d").astype("float64")
SC = lab.load("spot_close", "1d").astype("float64")
F = lab.load("fund", "1d").astype("float64")
BAS = (C / SC - 1).where(SC > 0)
# guard against obviously broken spot prints (|basis| > 5% -> nan)
BAS = BAS.where(BAS.abs() < 0.05)

def rm(x, L):
    return x.rolling(L, min_periods=max(1, L // 2 + 1)).mean() if L > 1 else x

# ---------------- diagnostics: rank IC by horizon ----------------
print("=== Rank IC diagnostics (contrarian score vs fwd h-day return), mean IC / t ===")
fw = {h: (C.shift(-h) / C - 1) for h in (1, 3, 7)}
diag = {"pm_lvl_1": -PM, "pm_lvl_7": -rm(PM, 7), "pm_z_7": -H.ts_z(rm(PM, 7), 90),
        "basis_lvl_1": -BAS, "basis_lvl_7": -rm(BAS, 7), "fund_7": -F.rolling(7, min_periods=4).sum()}
for k, s in diag.items():
    row = []
    for h, f in fw.items():
        ic = H.rank_ic(s, f, U)
        # overlapping horizons -> crude t with sqrt(n/h)
        row.append(f"h{h}: {ic.mean():+.4f} (t~{ic.mean() / ic.std() * np.sqrt(len(ic) / h):+.1f})")
    print(f"{k:14s}", "  ".join(row))

# XS correlation of premium signals with funding factor
f7 = F.rolling(7, min_periods=4).sum()
for k, s in {"pm_lvl_7": rm(PM, 7), "pm_lvl_1": PM, "basis_lvl_7": rm(BAS, 7), "pm_z_7": H.ts_z(rm(PM, 7), 90)}.items():
    a = s.where(U).rank(axis=1); b = f7.where(U & s.notna()).rank(axis=1); a = a.where(b.notna())
    ac = a.sub(a.mean(axis=1), axis=0); bc = b.sub(b.mean(axis=1), axis=0)
    cc = (ac * bc).sum(axis=1) / np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1))
    print(f"mean daily XS spearman({k}, fund_7) = {cc.mean():+.3f}")

# ---------------- cross-sectional configs ----------------
rows = []
rets = {}
def go(score, name, weights="q", rebal=1, note=""):
    if weights == "q":
        W = lab.quantile_ls_weights(score, U, q=0.2)
    else:
        W = lab.xs_demean_weights(score, U)
    o, res = H.evaluate(W, name, "A_basis", note=note, rebalance_every=rebal)
    print(H.fmt(o)); rows.append(o); rets[name] = res["df"]["r"]
    return o, res

print("\n=== A0 control: trailing funding factor ===")
o_f, res_f = go(-f7, "A0_fund7_ctrl")
fund_ctrl_r = res_f["df"]["r"]

print("\n=== A1 prem_mean level ===")
for L in (1, 3, 7, 14):
    go(-rm(PM, L), f"A1_pm_lvl_{L}")
print("\n=== A2 prem_mean trailing z ===")
for L in (1, 3, 7):
    go(-((rm(PM, L) - PM.rolling(90, min_periods=30).mean()) / PM.rolling(90, min_periods=30).std()), f"A2_pm_z_{L}")
print("\n=== A3 spot-perp basis level ===")
for L in (1, 3, 7):
    go(-rm(BAS, L), f"A3_basis_lvl_{L}")
print("\n=== A4 basis trailing z ===")
for L in (3, 7):
    go(-((rm(BAS, L) - BAS.rolling(90, min_periods=30).mean()) / BAS.rolling(90, min_periods=30).std()), f"A4_basis_z_{L}")
print("\n=== A5 premium beyond funding ===")
for L in (3, 7):
    rp = lab.xs_rank(rm(PM, L), U); rf = lab.xs_rank(F.rolling(L, min_periods=1).sum(), U)
    go(-(rp - rf), f"A5_pm_xfund_{L}")
print("\n=== A6 demean weights / A7 rebalance 3 ===")
z7 = -((rm(PM, 7) - PM.rolling(90, min_periods=30).mean()) / PM.rolling(90, min_periods=30).std())
go(-rm(PM, 7), "A6_pm_lvl_7_demean", weights="d")
go(z7, "A6_pm_z_7_demean", weights="d")
go(-rm(PM, 7), "A7_pm_lvl_7_rebal3", rebal=3)
go(z7, "A7_pm_z_7_rebal3", rebal=3)

# correlation of each XS strategy's returns with funding control
reg = pd.read_csv(H.REG)

# ---------------- time-series configs ----------------
print("\n=== A-TS time series (BTC / ETH / EW basket) ===")
EWpm = PM.where(U).mean(axis=1)
assets = {"BTC": ("BTCUSDT", PM["BTCUSDT"]), "ETH": ("ETHUSDT", PM["ETHUSDT"]), "EW": (None, EWpm)}
bh = {}
for a, (sym, s) in assets.items():
    for L in (3, 7):
        z = H.ts_z(rm(s, L), 90)
        for rule in ("a", "b"):
            pos = (-(z / 2).clip(-1, 1)) if rule == "a" else ((-z / 2).clip(0, 1))
            if sym is not None:
                W = pd.DataFrame(0.0, index=C.index, columns=C.columns); W[sym] = pos.fillna(0.0)
            else:
                n = U.sum(axis=1).replace(0, np.nan)
                W = U.astype(float).div(n, axis=0).mul(pos.fillna(0.0), axis=0).fillna(0.0)
            o, rr = H.evaluate(W, f"ATS_{a}_L{L}_{rule}", "A_basis_TS")
            print(H.fmt(o)); rows.append(o); rets[o["name"]] = rr["df"]["r"]
    # buy & hold reference (not a config, save=False)
    if sym is not None:
        W = pd.DataFrame(0.0, index=C.index, columns=C.columns); W[sym] = 1.0
    else:
        W = U.astype(float).div(U.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    o, _ = H.evaluate(W, f"ref_buyhold_{a}", "ref", save=False)
    print("   ref:", H.fmt(o))

# return-correlation with funding factor for every XS config
print("\n=== daily return corr with A0 funding control ===")
for o in rows:
    j = pd.concat([rets[o["name"]], fund_ctrl_r], axis=1).dropna()
    o["corr_fund_ctrl"] = float(j.iloc[:, 0].corr(j.iloc[:, 1]))
    print(f"{o['name']:30s} corr_fund_ctrl {o['corr_fund_ctrl']:+.2f}")
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(H.REG), "famA_results.csv"), index=False)
print("done")
