"""FAMILY B - OPEN-INTEREST DYNAMICS + TAKER FLOW (1d)

Hypotheses / rationale:
 B1 new leverage (OI growth) is crowding -> contrarian: coins where OI is building fast become the
    fuel for liquidation flushes; counterparties = market makers/basis traders paid to warehouse it.
 B2 price-OI quadrant 'squeeze fuel': price down + OI up = fresh shorts (squeeze fuel, expect bounce);
    price up + OI down = short covering (weak rally, expect fade). Continuous score =
    xsrank(dOI_k) - xsrank(ret_k). Must beat pure short-term reversal (-ret_k) to add anything.
 B3 leverage/crowding ratio OI / ADV30 (open interest relative to trading activity): high = crowded,
    illiquid leverage -> lower future returns / negative skew.  Direction: short high ratio.
 B4 taker imbalance (aggressive buy share of volume) is impatient (often retail/liquidation) flow ->
    temporary price pressure that reverts (liquidity-provision premium). Kline tbqv/qv is available
    from 2020, so this is the only family-B signal with a 2020-21 sample.
 B5 same with Binance metrics taker_ls_vol (2022-05 ->), consistency check.

Coverage (top-50 avg names/day): oi_usd 2021-12 -> 2024 (~39-50); taker_ls_vol 2022-05 ->;
tbqv/qv 2020 -> (14 names 2020, 35 in 2021).

PRE-REGISTERED GRID (written before running). Universe top-50 PIT, quintile L/S $-neutral, daily
rebalance, flat if < 20 valid names, cost 0.07%/side + funding.
 B1  -dlog(OI,k)                                 k in {1,3,7}      3
 B2  xsrank(dlogOI_k) - xsrank(ret_k)            k in {1,3,7}      3
 B2c -ret_k  (control: plain reversal)           k in {1,3,7}      3
 B2q quadrant: L = ret_k bottom tercile & dOI_k top tercile; S = ret_k top tercile & dOI_k bottom
     tercile, equal weight, $-neutral           k in {3,7}        2
 B3  -log(OI/ADV30) XS level; -tsz(log(OI/ADV30),90); -dlog(OI/ADV30,7)              3
 B4  -imb_k raw XS, imb_k = sum_k tbqv / sum_k qv  k in {1,3,7}     3
     -(imb_k - trailing 90d mean of imb_1)        k in {1,3,7}     3
 B5  -tsz(log taker_ls_vol, 30)                                    1
                                                       subtotal   21
 Robustness on best 2: reb3, top100, q33, funding-neutral          8
 Optional 4h version of best fast OI signal (k in {1,3,6} bars)    <=3
 TOTAL <= 32
"""
from h2lib import *

U = lab.universe("1d", top_k=50)
C = lab.load("close", "1d"); QV = lab.load("qv", "1d"); TB = lab.load("tbqv", "1d")
OI = lab.load("oi_usd", "1d"); OI = OI.where(OI > 0)
TL = lab.load("taker_ls_vol", "1d"); TL = TL.where(TL > 0)
F = lab.load("fund", "1d")
R1 = C.pct_change(fill_method=None).shift(-1); R7 = C.shift(-7) / C - 1
fund7 = F.rolling(7, min_periods=5).sum()
ADV30 = QV.rolling(30, min_periods=15).mean()
lev = np.log(OI / ADV30)
xr = lambda X: X.where(U).rank(axis=1, pct=True)

sig = {}
for k in (1, 3, 7):
    dOI = lchg(OI, k); rk = C / C.shift(k) - 1
    sig[f"B1_-dOI{k}"] = -dOI
    sig[f"B2_sq{k}"] = xr(dOI) - xr(rk)
    sig[f"B2c_-ret{k}"] = -rk
    imb = TB.rolling(k).sum() / QV.rolling(k).sum()
    sig[f"B4_-imb{k}"] = -imb
    imb1m = (TB / QV).rolling(90, min_periods=45).mean()
    sig[f"B4_-imbdm{k}"] = -(imb - imb1m)
sig["B3_-lev"] = -lev
sig["B3_-zlev90"] = -tsz(lev, 90)
sig["B3_-dlev7"] = -(lev - lev.shift(7))
sig["B5_-zTL30"] = -tsz(np.log(TL), 30)

print("Diagnostics: XS IC vs next-1d/7d return; xs corr with past ret7 and fund7")
diag = {}
for n, s in sig.items():
    ic1 = xs_ic(s, R1, U); ic7 = xs_ic(s, R7, U)
    diag[n] = {"IC1": ic1[0], "t1": ic1[1], "IC7": ic7[0], "t7": ic7[1], "days": ic1[2],
               "c_ret7": xs_corr(s, C / C.shift(7) - 1, U), "c_fund7": xs_corr(s, fund7, U)}
print(pd.DataFrame(diag).T.round(3).to_string())

rows = {}
for n, s in sig.items():
    row, res = evaluate(n, "B", qls(s, U)); rows[n] = row; print(fmt(row))

# B2q discrete quadrants
for k in (3, 7):
    dOI = lchg(OI, k); rk = C / C.shift(k) - 1
    ok = (dOI.where(U).notna() & rk.where(U).notna())
    enough = ok.sum(axis=1) >= 20
    pr = rk.where(ok).rank(axis=1, pct=True); po = dOI.where(ok).rank(axis=1, pct=True)
    L = ((pr <= 1/3) & (po >= 2/3)).astype(float); S = ((pr >= 2/3) & (po <= 1/3)).astype(float)
    L = L.where(np.broadcast_to(enough.values[:, None], L.shape), 0.0); S = S.where(np.broadcast_to(enough.values[:, None], S.shape), 0.0)
    W = L.div(L.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5 - S.div(S.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5
    n = f"B2q_quad{k}"
    row, res = evaluate(n, "B", W); rows[n] = row; print(fmt(row))
pd.DataFrame(rows).T.to_csv(os.path.join(HERE, "family_B_grid.csv"))
