"""FAMILY B - ORDER FLOW & ATTENTION (h3_basis_flow).  Research data 2020-2024 only via lab.py.

HYPOTHESES / RATIONALE
 B1/B2 taker-buy ratio TBR = tbqv/qv. Aggressive (taker) buying = informed or impatient demand.
   Continuation if takers are informed (Kyle); reversal if takers are noise/levered retail whose
   price pressure is absorbed by liquidity providers who get paid back (Grossman-Miller).
   Raw TBR differs structurally across coins (MM mix) -> also per-coin trailing z.
 B3/B4 attention: volume / trade-count shock vs trailing 30d. Attention-grabbing coins attract
   retail buying -> overreaction -> reversal; OR high-volume return premium (visibility).
 B5 shock-conditioned reversal: short-term reversal should be strongest where the move came with
   an attention shock (liquidity-driven, uninformed) -> score=-ret_rank * shock_rank. Plain
   reversal is run as control.
 B6 flow-price divergence: heavy taker buying with flat/negative price = passive sellers absorbing
   (informed supply) -> price falls next; heavy selling absorbed with price up -> rises.
   score = xsrank(ret_L) - xsrank(tbr_z_L).

PRE-REGISTERED GRID (universe lab.universe(tf, min_hist_days=60, top_k=50); quintile L/S q=0.2;
rebalance every bar):
 B1 tbr_lvl_L   TBR_L = sum(tbqv,L)/sum(qv,L); +TBR (cont) and -TBR (rev)   L in {1,3,7}      6
 B2 tbr_z_L     (TBR_L - mean60(TBR_1d))/std60(TBR_1d), trailing; both signs L in {1,3,7}      6
 B3 vshock_L    mean(qv,L)/mean(qv,30).shift(L); both signs                  L in {1,3}        4
 B4 tshock_L    same with trades; both signs                                 L in {1,3}        4
 B5 rev_L       control -ret_L, and shock-cond. -(xsrank(ret_L)-.5)*xsrank(vshock_L) L in {1,3} 4
 B6 div_L       xsrank(ret_L) - xsrank(tbr_z_L)                              L in {1,3,7}      3
 B7 4h tbr_z    4h bars: TBR over L bars, z vs trailing 360 bars; both signs L in {1,6} bars   4
 TOTAL family B = 31 configs. Diagnostics: rank IC h=1,3,7d (1d signals).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import h3lib as H
import lab

U = lab.universe("1d", min_hist_days=60, top_k=50)
C = lab.load("close", "1d").astype("float64")
QV = lab.load("qv", "1d").astype("float64")
TB = lab.load("tbqv", "1d").astype("float64")
TR = lab.load("trades", "1d").astype("float64")
QV = QV.where(QV > 0); TR = TR.where(TR > 0)
_r = TB / QV
TBR1 = _r.where((_r > 0.05) & (_r < 0.95))


def rs(x, L):
    return x.rolling(L, min_periods=L).sum() if L > 1 else x


def tbr(L):
    return (rs(TB, L) / rs(QV, L)).where(TBR1.notna())


def tbr_z(L):
    return (tbr(L) - TBR1.rolling(60, min_periods=30).mean()) / TBR1.rolling(60, min_periods=30).std()


def shock(X, L):
    return X.rolling(L, min_periods=L).mean() / X.rolling(30, min_periods=20).mean().shift(L)


def ret(L):
    return C / C.shift(L) - 1


print("=== Rank IC diagnostics (score as defined, vs fwd h-day return) ===")
fw = {h: (C.shift(-h) / C - 1) for h in (1, 3, 7)}
diag = {"tbr_lvl_1": tbr(1), "tbr_lvl_7": tbr(7), "tbr_z_1": tbr_z(1), "tbr_z_7": tbr_z(7),
        "vshock_1": shock(QV, 1), "vshock_3": shock(QV, 3), "tshock_1": shock(TR, 1),
        "rev_1": -ret(1), "rev_3": -ret(3),
        "div_1": lab.xs_rank(ret(1), U) - lab.xs_rank(tbr_z(1), U),
        "div_7": lab.xs_rank(ret(7), U) - lab.xs_rank(tbr_z(7), U)}
for k, s in diag.items():
    row = []
    for h, f in fw.items():
        ic = H.rank_ic(s, f, U)
        row.append(f"h{h}: {ic.mean():+.4f} (t~{ic.mean() / ic.std() * np.sqrt(len(ic) / h):+.1f})")
    print(f"{k:12s}", "  ".join(row))
# XS correlation of flow signals with own return (is TBR just a return proxy?)
for k in ("tbr_z_1", "tbr_z_7"):
    L = int(k[-1])
    a = diag[k].where(U).rank(axis=1); b = ret(L).where(U & diag[k].notna()).rank(axis=1); a = a.where(b.notna())
    ac = a.sub(a.mean(axis=1), axis=0); bc = b.sub(b.mean(axis=1), axis=0)
    print(f"mean XS spearman({k}, ret_{L}) = {((ac*bc).sum(axis=1)/np.sqrt((ac**2).sum(axis=1)*(bc**2).sum(axis=1))).mean():+.3f}")

rows = []
def go(score, name, Uu=U, tf="1d"):
    W = lab.quantile_ls_weights(score, Uu, q=0.2)
    o, res = H.evaluate(W, name, "B_flow", tf=tf)
    print(H.fmt(o)); rows.append(o)
    return o

print("\n=== B1 taker-buy ratio level ===")
for L in (1, 3, 7):
    s = tbr(L); go(s, f"B1_tbr_lvl_{L}_cont"); go(-s, f"B1_tbr_lvl_{L}_rev")
print("\n=== B2 taker-buy ratio trailing z ===")
for L in (1, 3, 7):
    s = tbr_z(L); go(s, f"B2_tbr_z_{L}_cont"); go(-s, f"B2_tbr_z_{L}_rev")
print("\n=== B3 volume shock ===")
for L in (1, 3):
    s = shock(QV, L); go(s, f"B3_vshock_{L}_long"); go(-s, f"B3_vshock_{L}_short")
print("\n=== B4 trade-count shock ===")
for L in (1, 3):
    s = shock(TR, L); go(s, f"B4_tshock_{L}_long"); go(-s, f"B4_tshock_{L}_short")
print("\n=== B5 reversal control / shock-conditioned reversal ===")
for L in (1, 3):
    go(-ret(L), f"B5_rev_{L}_ctrl")
    go(-(lab.xs_rank(ret(L), U) - 0.5) * lab.xs_rank(shock(QV, L), U), f"B5_shockrev_{L}")
print("\n=== B6 flow-price divergence ===")
for L in (1, 3, 7):
    go(lab.xs_rank(ret(L), U) - lab.xs_rank(tbr_z(L), U), f"B6_div_{L}")

pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(H.REG), "famB_results_1d.csv"), index=False)

print("\n=== B7 4h taker-buy z ===")
lab.clear()
U4 = lab.universe("4h", min_hist_days=60, top_k=50)
QV4 = lab.load("qv", "4h").astype("float64"); TB4 = lab.load("tbqv", "4h").astype("float64")
QV4 = QV4.where(QV4 > 0)
_r4 = TB4 / QV4
T1 = _r4.where((_r4 > 0.05) & (_r4 < 0.95))
mu = T1.rolling(360, min_periods=180).mean(); sd = T1.rolling(360, min_periods=180).std()
for L in (1, 6):
    tL = (TB4.rolling(L, min_periods=L).sum() / QV4.rolling(L, min_periods=L).sum()).where(T1.notna()) if L > 1 else T1
    z = (tL - mu) / sd
    go(z, f"B7_4h_tbr_z_{L}_cont", Uu=U4, tf="4h"); go(-z, f"B7_4h_tbr_z_{L}_rev", Uu=U4, tf="4h")
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(H.REG), "famB_results.csv"), index=False)
print("done")
