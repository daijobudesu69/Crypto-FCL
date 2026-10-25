"""FAMILY B(i) - delta-neutral cash-and-carry (long Binance spot + short USDT perp), 1d, 2020-2024 IS.

Per unit notional on coin i over bar t+1:  pnl = spot_ret - perp_ret + fund[t+1]
(basis change = spot_ret - perp_ret; funding received by the short perp).
Costs: spot taker 0.10%/side + perp 0.07%/side on |change in notional| (both legs traded together).
Weights = notional per coin as fraction of the SPOT capital (sum <= 1). Perp margin extra:
returns are also shown on total capital assuming perp margin = notional/3 (capital = 1.33x notional).
Hedge drift between rebalances ignored (legs move together; error tiny).

PRE-REGISTERED GRID (written before running):
 signal = trailing L-day funding average, annualised (sum/L*365); enter coins with signal > thr
 B1 universe top50 perp liquidity with spot close available at t:  L in {3,7} x thr in {0,10%,20%,40%}  -> 8
 B2 BTC+ETH only:                                                   L in {3,7} x thr in {0,10%,20%,40%}  -> 8
 B3 always-on BTC+ETH (no filter) benchmark                                                              -> 1
 B4 top50, hold top-N by signal (if signal>0): N in {3,10}, L=7                                          -> 2
 weighting: equal notional across selected coins (sum = 1), else cash (0%).   rebalance daily.
 TOTAL = 19 configs.
"""
from common import *
F = lab.load("fund", "1d"); P = lab.load("close", "1d"); S = lab.load("spot_close", "1d")
U = lab.universe("1d", top_k=50)
rp = P.pct_change(fill_method=None); rs = S.pct_change(fill_method=None)
nxt_ok = rp.shift(-1).notna() & rs.shift(-1).notna()
basis_n = (rs - rp).shift(-1)   # basis pnl over t+1
fund_n = F.shift(-1)
SC, PC = 0.0010, 0.0007
print("daily basis pnl dist (in U, spot avail):")
print(pd.Series((rs - rp).where(U).values.ravel()).dropna().describe(percentiles=[.001, .01, .5, .99, .999]))


def run(Wt, name):
    Wt = Wt.fillna(0.0)
    Wt = Wt.where(nxt_ok, 0.0)
    b = (Wt * basis_n.fillna(0)).sum(axis=1)
    f = (Wt * fund_n.fillna(0)).sum(axis=1)
    turn = Wt.diff().abs().sum(axis=1)
    turn.iloc[0] = Wt.iloc[0].abs().sum()
    c = turn * (SC + PC)
    r = b + f - c
    r.index = r.index + pd.Timedelta("1D")
    b.index = r.index; f.index = r.index; c.index = r.index
    expo = Wt.sum(axis=1); expo.index = r.index
    nz = np.flatnonzero(expo.values > 0)
    first = r.index[0]
    r = r.iloc[:-1]
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    cap = r / (4 / 3)
    eqc = (1 + cap).cumprod()
    yr = r.groupby(r.index.year)
    d = {"name": name, "apr_notional": eq.iloc[-1] ** (1 / yrs) - 1, "apr_capital": eqc.iloc[-1] ** (1 / yrs) - 1,
         "sharpe": sh(r), "maxdd": float((eq / eq.cummax() - 1).min()), "fund_yr": f.mean() * 365,
         "basis_yr": b.mean() * 365, "cost_yr": c.mean() * 365, "avg_expo": expo.mean(), "turn_yr": turn.mean() * 365,
         "avg_n": (Wt > 0).sum(axis=1).mean(), "worst_day": r.min()}
    ylog = yr.apply(lambda x: np.log1p(x).sum())
    d["share_2021"] = float(ylog.get(2021, 0) / ylog.sum()) if ylog.sum() != 0 else np.nan
    for y in range(2020, 2025):
        d[f"ret_{y}"] = float(np.expm1(ylog.get(y, 0)))
    d.update({"sh_" + k: v for k, v in subperiods(r).items()})
    return d, r


rows = []
elig = U & S.notna() & P.notna()
for L in [3, 7]:
    sig = F.rolling(L, min_periods=L).sum() / L * 365
    for thr in [0.0, 0.10, 0.20, 0.40]:
        sel = elig & (sig > thr)
        Wt = sel.astype(float).div(sel.sum(axis=1).replace(0, np.nan), axis=0)
        d, _ = run(Wt, f"B1_top50_L{L}_thr{int(thr*100)}"); rows.append(d)
        sel2 = sel.copy(); sel2.loc[:, [c for c in sel.columns if c not in ("BTCUSDT", "ETHUSDT")]] = False
        Wt2 = sel2.astype(float).div(sel2.sum(axis=1).replace(0, np.nan), axis=0)
        d, _ = run(Wt2, f"B2_btceth_L{L}_thr{int(thr*100)}"); rows.append(d)
sel = elig.copy(); sel.loc[:, [c for c in sel.columns if c not in ("BTCUSDT", "ETHUSDT")]] = False
d, rb3 = run(sel.astype(float).div(sel.sum(axis=1).replace(0, np.nan), axis=0), "B3_btceth_always"); rows.append(d)
sig = F.rolling(7, min_periods=7).sum() / 7 * 365
for N in [3, 10]:
    rk = sig.where(elig & (sig > 0)).rank(axis=1, ascending=False)
    sel = rk <= N
    Wt = sel.astype(float).div(sel.sum(axis=1).replace(0, np.nan), axis=0)
    d, _ = run(Wt, f"B4_top{N}_L7"); rows.append(d)
out = pd.DataFrame(rows)
out.to_csv("famB_carry.csv", index=False)
pd.set_option("display.width", 260)
print(out[["name", "apr_notional", "apr_capital", "sharpe", "maxdd", "fund_yr", "basis_yr", "cost_yr", "avg_expo", "avg_n",
           "turn_yr", "share_2021", "ret_2020", "ret_2021", "ret_2022", "ret_2023", "ret_2024", "worst_day"]].round(3).to_string())
print(out[["name", "sh_2020-21", "sh_2022", "sh_2023-24"]].round(2).to_string())
