"""Step 3: survivorship. Universe variants for SIGNAL and/or TRADED alt basket:
 (a) coins first listed (Binance perp) before 2022-01-01 only
 (b) large legacy majors only: L14 = 14 pre-2020-03 alts on HYPE (hunter's list); L8 = ETH XRP BCH LTC BNB ADA LINK TRX
 (c) coins in their first 180 days excluded
For each: Sharpe, sub-periods, LONG-SPREAD vs SHORT-SPREAD state returns, alt-leg vs BTC-leg."""
import numpy as np, pandas as pd
import dc
C = dc.C; BTC = dc.BTC
U = dc.U50()
cols = pd.Series(True, index=C.columns)
pre22 = pd.Series(dc.FIRST < pd.Timestamp("2022-01-01", tz="UTC"), index=C.columns)
L14 = [s for s in dc.FIRST.index[dc.FIRST < pd.Timestamp("2020-03-01", tz="UTC")] if s in dc.lab.hype_symbols() and s != BTC]
L8 = ["ETHUSDT", "XRPUSDT", "BCHUSDT", "LTCUSDT", "BNBUSDT", "ADAUSDT", "LINKUSDT", "TRXUSDT"]
base_elig = C.notna() & (dc.HIST >= 60)
def mask_cols(lst):
    return pd.Series(C.columns.isin(lst + [BTC]), index=C.columns)
U_pre22 = U & pre22
U_L14 = base_elig & mask_cols(L14)
U_L8 = base_elig & mask_cols(L8)
U_old180 = U & (dc.HIST >= 180)
st_base = dc.disp7_state(U)

def run(name, sig_U, trade_U):
    st = st_base if sig_U is U else dc.disp7_state(sig_U)
    W = dc.spread_w(st, trade_U)
    r = dc.bt(W)["df"]["r"]
    sa = dc.align_state(st, r.index)
    Wa = W.copy(); Wa[BTC] = 0.0; Wb = W * 0.0; Wb[BTC] = W[BTC]
    ra = dc.bt(Wa)["df"]["r"]; rb = dc.bt(Wb)["df"]["r"]
    agree = float((st == st_base).where(st.notna() & st_base.notna()).mean())
    s1 = r[sa == 1]; s_1 = r[sa == -1]
    sb = dc.subs(r)
    return (f"| {name} | {dc.sharpe(r):.2f} | {sb['20-21']:.2f} / {sb['22']:.2f} / {sb['23-24']:.2f} | {dc.sharpe(r[r.index.year == 2024]):.2f} | "
            f"{s1.mean() * 365:+.0%} ({dc.sharpe(s1):.2f}) | {s_1.mean() * 365:+.0%} ({dc.sharpe(s_1):.2f}) | {ra.mean() * 365:+.0%} | {rb.mean() * 365:+.0%} | "
            f"{agree:.2f} | {trade_U.sum(axis=1)[r.index].mean():.0f} |"), r

L = ["## Step 3 — survivorship",
     f"L14 = {', '.join(x.replace('USDT', '') for x in L14)}; L8 = {', '.join(x.replace('USDT', '') for x in L8)}. "
     f"Coins in HYPE top-50 that were listed before 2022: {int((U & pre22).any().sum())} of {int(U.any().sum())} ever-members.",
     "", "| variant (signal universe -> traded universe) | Sharpe | 20-21 / 22 / 23-24 | 2024 | LONG-SPREAD ann (SR) | SHORT-SPREAD ann (SR) | alt-leg ann | BTC-leg ann | state agree w/ base | avg names |",
     "|---|---|---|---|---|---|---|---|---|---|"]
rows = {}
for nm, su, tu in (("base U50 -> U50", U, U),
                   ("(a) U50 -> pre-2022 listings", U, U_pre22), ("(a) pre-2022 -> pre-2022", U_pre22, U_pre22),
                   ("(b) U50 -> L14 majors", U, U_L14), ("(b) L14 -> L14", U_L14, U_L14),
                   ("(b) U50 -> L8 majors", U, U_L8), ("(b) L8 -> L8", U_L8, U_L8),
                   ("(c) U50 -> ex first-180d", U, U_old180), ("(c) ex-180d -> ex-180d", U_old180, U_old180)):
    line, r = run(nm, su, tu); L.append(line); rows[nm] = r
    print(line, flush=True)
# contribution of young coins (first 180d / first 365d) to the base alt leg, by state
Wb = dc.spread_w(st_base, U); Wb[BTC] = 0.0
Rn = dc.R.shift(-1).fillna(0.0)
pnl = Wb * Rn
young180 = dc.HIST < 180; young365 = dc.HIST < 365
s = st_base.reindex(C.index)
for sv in (1, -1):
    m = (s == sv)
    tot = pnl[m].sum().sum()
    y1 = pnl[m].where(young180[m]).sum().sum(); y3 = pnl[m].where(young365[m]).sum().sum()
    w1 = Wb[m].abs().where(young180[m]).sum().sum() / Wb[m].abs().sum().sum()
    L.append(f"Base alt leg in state {sv:+d}: total arithmetic P&L {tot:+.1%}; from coins in their first 180d {y1:+.1%} (weight share {w1:.0%}), first 365d {y3:+.1%}.")
post = ~pre22
for sv in (1, -1):
    m = (s == sv)
    L.append(f"  state {sv:+d}: P&L from coins listed 2022+ {pnl[m].loc[:, post].sum().sum():+.1%}")
# which coins carry the alt leg in LONG-SPREAD state
m = (s == 1)
c1 = pnl[m].sum().sort_values()
L.append("Top-8 alt contributors in LONG-SPREAD state: " + ", ".join(f"{k.replace('USDT', '')} {v:+.1%}" for k, v in c1.tail(8)[::-1].items())
         + f" | sum all {c1.sum():+.1%}, ex top-5 {c1.iloc[:-5].sum():+.1%}")
m = (s == -1)
c2 = pnl[m].sum().sort_values()
L.append("Worst-5 alt contributors in SHORT-SPREAD state (short survivors that rallied): " + ", ".join(f"{k.replace('USDT', '')} {v:+.1%}" for k, v in c2.head(5).items())
         + f" | best-5: " + ", ".join(f"{k.replace('USDT', '')} {v:+.1%}" for k, v in c2.tail(5)[::-1].items()) + f" | sum {c2.sum():+.1%}")
dc.append("\n".join(L))
