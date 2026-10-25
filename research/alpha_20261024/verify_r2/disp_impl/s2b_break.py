"""disp_impl step 2b - break attempts on the replicated DISP7 (minp=120), counted as configs:
 T1..T4 threshold neighbours (hi/lo) 0.60/0.40, 0.70/0.30, 0.75/0.25, 0.55/0.45      (4 configs)
 G1 forced-flat realism: keep weight through NaN next-bar (earn 0) instead of lab forced exit (1 config)
 Episode concentration: PnL by state spell; share of total log-PnL from top 3 spells; drop-top-k Sharpe
 Common interior gap dates.
Outputs -> s2b_out.txt
"""
import sys, os
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lab

OUT = os.path.dirname(os.path.abspath(__file__))
log = open(os.path.join(OUT, "s2b_out.txt"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.write(s + "\n"); log.flush()


BTC = "BTCUSDT"
C = lab.load("close", "1d")
U = lab.hype_universe("1d", top_k=50)
disp = np.log(C / C.shift(7)).where(U).std(axis=1)
p = disp.rolling(365, min_periods=120).rank(pct=True, method="max")


def state(hi, lo):
    st = np.zeros(len(p)); cur = 0.0
    for i, v in enumerate(p.values):
        if np.isfinite(v):
            if v > hi: cur = 1.0
            elif v < lo: cur = -1.0
        st[i] = cur
    return pd.Series(st, index=p.index)


def W_of(st):
    alts = U.copy(); alts[BTC] = False
    W = alts.astype(float).div(alts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).mul(st, axis=0) * 0.5
    W[BTC] = -0.5 * st
    return W


def sh(r):
    r = r.dropna(); return r.mean() / r.std() * np.sqrt(365)


def subs(r):
    return {k: round(sh(r[(r.index >= pd.Timestamp(a, tz='UTC')) & (r.index < pd.Timestamp(b, tz='UTC'))]), 2)
            for k, a, b in [("20-21", "2020-01-01", "2022-01-01"), ("22", "2022-01-01", "2023-01-01"), ("23-24", "2023-01-01", "2025-01-01")]}


base_st = state(2 / 3, 1 / 3)
base = lab.backtest(W_of(base_st), "1d", cost=lab.HYPE_COST)["df"]
P("base Sharpe", round(sh(base["r"]), 3), subs(base["r"]))
for hi, lo in [(0.60, 0.40), (0.70, 0.30), (0.75, 0.25), (0.55, 0.45)]:
    st = state(hi, lo)
    r = lab.backtest(W_of(st), "1d", cost=lab.HYPE_COST)["df"]["r"]
    y = lab.yearly(r)["ret"]
    P(f"T hi/lo {hi}/{lo}: Sharpe {sh(r):.3f} subs {subs(r)} switches {int((st.diff().abs()>0).sum())} yearly {dict((int(k), round(v,3)) for k,v in y.items())}")

# G1: keep weight through gaps (no forced flat), earn 0 on NaN bar
W = W_of(base_st)
R = C.pct_change(fill_method=None)
Rn = R.shift(-1).fillna(0.0).values
Fn = lab.load("fund", "1d").shift(-1).fillna(0.0).values
w = W.reindex(columns=C.columns).fillna(0.0).values
g = (w * Rn).sum(1); f = -(w * Fn).sum(1)
prev = np.vstack([np.zeros(w.shape[1]), w[:-1]]); Rc = np.vstack([np.zeros(w.shape[1]), Rn[:-1]])
pp = np.concatenate([[0.0], g[:-1]])
turn = np.abs(w - prev * (1 + Rc) / (1 + pp)[:, None]).sum(1)
rg = pd.Series(g + f - turn * lab.HYPE_COST, index=C.index + pd.Timedelta("1D"))[:-1]
rg = rg[rg.index >= base.index[0]]
P(f"G1 no-forced-flat: Sharpe {sh(rg):.3f} (base {sh(base['r']):.3f})")

# common gap dates
gapmask = C.isna() & C.ffill().notna() & C.bfill().notna()
gd = gapmask.sum(axis=1); P("interior-gap dates with >=5 coins missing:", [str(d.date()) for d in gd[gd >= 5].index])

# per-state and episode concentration (realised returns, index = realisation day; state known the day before)
r = base["r"]
st_r = base_st.shift(1).reindex(r.index)  # state that generated return realised at index
for s in [1, -1]:
    x = r[st_r == s]
    P(f"state {s:+.0f}: days {len(x)} ann ret {x.mean()*365:.1%} SR {sh(x):.2f}")
spell = (st_r != st_r.shift()).cumsum()
ep = pd.DataFrame({"lr": np.log1p(r), "st": st_r, "sp": spell}).groupby("sp").agg(lr=("lr", "sum"), st=("st", "first"), n=("lr", "size"),
                                                                                    start=("lr", lambda z: z.index[0]))
ep = ep[ep["st"] != 0]
tot = ep["lr"].sum()
srt = ep.sort_values("lr", ascending=False)
P(f"episodes {len(ep)}; total logPnL {tot:.3f}; top1 {srt['lr'].iloc[0]:.3f} top3 {srt['lr'].iloc[:3].sum():.3f} ({srt['lr'].iloc[:3].sum()/tot:.0%} of total); "
  f"positive episodes {(ep['lr']>0).mean():.0%}")
P("top 5 episodes:\n" + srt.head(5).to_string())
P("worst 3 episodes:\n" + srt.tail(3).to_string())
for k in [1, 3, 5]:
    drop = srt.index[:k]
    keep = ~spell.isin(drop)
    P(f"Sharpe dropping top-{k} episodes' days: {sh(r[keep.values]):.3f}")
log.close()
