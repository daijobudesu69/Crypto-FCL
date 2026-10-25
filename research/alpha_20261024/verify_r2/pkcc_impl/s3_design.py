"""Step 3: $200 design. PRE-REGISTERED grid (see REPORT.md 'Step 3 PRE-REGISTRATION', written before this run):
 common: U = hype_universe top50 & ratio valid, ratio n30 min24, long lowest / short highest ratio, EW G/(2k),
         k = min(floor(E*G/2/12.5), kmax, floor(N/4)); k<3 -> flat. account_sim mechanics, $200, $10 min order.
 D1 G1.00 k8 | D2 G1.25 k8 | D3 G1.5 k8 | D4 G1.0 k6 | D5 G1.25 k6 | D6 G1.5 k6 | D7 G1.25 kq | D8 G1.25 k8 band3
 D9 G1.25 k6 band3 | D10 G1.25 k8 reb2 | D11 G1.25 k8 reb7 | D12 G1.0 kq      (kq = floor(0.2N))
 runs per config: full 2020-03-01.., fresh 2023-01-01.., lag1 full (stress).
 selection: max of min(SR_full, SR_2023, SR_lag1) with final equity > 200 in full & 2023 and equity never < 80;
            ties within 0.05 -> lower MaxDD full, then fewer orders/day.
"""
import json, time
import numpy as np, pandas as pd
from common_pk import *
import account_sim

H, L, C = panels()
U = lab.hype_universe("1d", top_k=50)
S = ratio_signal(H, L, C, 30, 24)
M = U & S.notna()
Cl = lab.load("close", "1d")
Rn = Cl.pct_change(fill_method=None).shift(-1).values.astype("float64")
Fn = lab.load("fund", "1d").shift(-1).values.astype("float64")
IX = Cl.index; COLS = Cl.columns
COST = lab.HYPE_COST

CFG = {"D1": (1.00, 8, 0, 1), "D2": (1.25, 8, 0, 1), "D3": (1.50, 8, 0, 1), "D4": (1.00, 6, 0, 1),
       "D5": (1.25, 6, 0, 1), "D6": (1.50, 6, 0, 1), "D7": (1.25, "q", 0, 1), "D8": (1.25, 8, 3, 1),
       "D9": (1.25, 6, 3, 1), "D10": (1.25, 8, 0, 2), "D11": (1.25, 8, 0, 7), "D12": (1.00, "q", 0, 1)}


def dyn(G, kmax, band, reb, lag=0, start="2020-03-01", equity0=200.0, min_order=10.0, min_trade=10.0, per_name=12.5):
    Sv = S.shift(lag).values if lag else S.values
    Mv = M.shift(lag, fill_value=False).values if lag else M.values
    i0 = int(np.searchsorted(IX, pd.Timestamp(start, tz="UTC")))
    nC = len(COLS)
    pos = np.zeros(nC); eq = equity0
    w = np.zeros(nC); Lset = []; Sset = []
    Wrec = np.zeros((len(IX), nC))
    eqs = []; orders = 0; fees = 0.0; funds = 0.0; skipped = 0; ks = []
    for n, i in enumerate(range(i0, len(IX))):
        if n % reb == 0:
            valid = Mv[i] & ~np.isnan(Sv[i])
            idx = np.flatnonzero(valid)
            N = len(idx)
            kcap = int(np.floor(0.2 * N)) if kmax == "q" else kmax
            k = min(int(np.floor(eq * G / 2 / per_name)), kcap, N // 4)
            w = np.zeros(nC)
            if k >= 3:
                asc = idx[np.argsort(Sv[i][idx], kind="mergesort")]
                if band:
                    rk = {j: r for r, j in enumerate(asc)}           # 0 = lowest ratio
                    keepL = sorted([j for j in Lset if j in rk and rk[j] < k + band], key=lambda j: rk[j])[:k]
                    for j in asc:
                        if len(keepL) >= k: break
                        if j not in keepL: keepL.append(j)
                    keepS = sorted([j for j in Sset if j in rk and (N - 1 - rk[j]) < k + band], key=lambda j: -rk[j])[:k]
                    for j in asc[::-1]:
                        if len(keepS) >= k: break
                        if j not in keepS and j not in keepL: keepS.append(j)
                    Lset, Sset = keepL, keepS
                else:
                    Lset, Sset = list(asc[:k]), list(asc[-k:])
                w[Lset] = G / (2 * k); w[Sset] = -G / (2 * k)
            else:
                Lset, Sset = [], []
            ks.append(k)
        Wrec[i] = w
        # ---- account_sim mechanics (copied semantics)
        r = Rn[i]; f = Fn[i]
        tgt = w * eq
        small = (np.abs(tgt) < min_order) & (tgt != 0)
        skipped += int(small.sum()); tgt[small] = 0.0
        tgt[np.isnan(r)] = 0.0
        diff = tgt - pos
        trade = (np.abs(diff) >= min_trade) | ((tgt == 0) & (pos != 0))
        new_pos = np.where(trade, tgt, pos)
        traded = np.abs(new_pos - pos).sum()
        orders += int((np.abs(new_pos - pos) > 0).sum())
        fee = traded * COST
        r0 = np.nan_to_num(r); f0 = np.nan_to_num(f)
        pnl = (new_pos * r0).sum(); fund = -(new_pos * f0).sum()
        eq = eq + pnl + fund - fee
        fees += fee; funds += fund
        pos = new_pos * (1 + r0)
        eqs.append(eq)
        if eq <= 0: break
    e = pd.Series(eqs, index=IX[i0:i0 + len(eqs)] + pd.Timedelta("1D"))
    Wdf = pd.DataFrame(Wrec, index=IX, columns=COLS)
    return e, Wdf, {"orders": orders, "fees": fees, "funding": funds, "skipped": skipped,
                    "k_mean": float(np.mean(ks)) if ks else 0, "k_at_start": ks[0] if ks else 0}


def stats(e, eq0=200.0):
    r = pd.concat([pd.Series([e.iloc[0] / eq0 - 1], index=[e.index[0]]), e.pct_change().iloc[1:]])
    yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    return {"sr": sharpe(r), "final": float(e.iloc[-1]), "mdd": float((e / e.cummax() - 1).min()),
            "min_eq": float(e.min()), "cagr": float((e.iloc[-1] / eq0) ** (365 / len(e)) - 1),
            "yrs": {int(k): round(float(v), 3) for k, v in yr.items()}, "sub": subp(r)}


rows = {}
lines = []
for cid, (G, kmax, band, reb) in (CFG.items() if __name__ == "__main__" else []):
    t0 = time.time()
    out = {"G": G, "kmax": kmax, "band": band, "reb": reb}
    for run, kw in (("full", {}), ("y23", {"start": "2023-01-01"}), ("lag1", {"lag": 1})):
        e, Wd, info = dyn(G, kmax, band, reb, **kw)
        st = stats(e); st.update(info)
        days = len(e)
        st["orders_per_day"] = info["orders"] / days
        if run in ("full", "y23"):
            # replay through account_sim.simulate (official small-account simulator)
            st_ = kw.get("start", "2020-03-01")
            o, s = account_sim.simulate(Wd, "1d", equity0=200, min_order=10, cost=COST, start=st_)
            st["replay_final"] = s["equity_end"]; st["replay_maxdd"] = s["maxdd"]
            st["replay_match"] = bool(abs(s["equity_end"] - st["final"]) < 1e-6 * max(1, st["final"]))
            if run == "full":
                fr = lab.backtest(Wd, "1d", cost=COST)["metrics"]
                st["frictionless_sr"] = fr["sharpe"]
        out[run] = st
    out["score"] = min(out["full"]["sr"], out["y23"]["sr"], out["lag1"]["sr"])
    out["eligible"] = bool(out["full"]["final"] > 200 and out["y23"]["final"] > 200 and out["full"]["min_eq"] >= 80
                           and out["y23"]["min_eq"] >= 80)
    rows[cid] = out
    f, y, l = out["full"], out["y23"], out["lag1"]
    s = (f"{cid:4s} G{G} k{kmax} b{band} r{reb} | FULL SR {f['sr']:.2f} (fricless {f['frictionless_sr']:.2f}) "
         f"$200->{f['final']:.0f} MDD {f['mdd']:.0%} minEq {f['min_eq']:.0f} sub {f['sub']['2020-21']:.2f}/{f['sub']['2022']:.2f}/{f['sub']['2023-24']:.2f} "
         f"yrs {' '.join(f'{k}:{v:+.0%}' for k, v in f['yrs'].items())} ord/d {f['orders_per_day']:.1f} fees ${f['fees']:.0f} fund ${f['funding']:+.0f} skip {f['skipped']} kmean {f['k_mean']:.1f} replay {f['replay_match']}"
         f" | Y23 SR {y['sr']:.2f} $200->{y['final']:.0f} MDD {y['mdd']:.0%} minEq {y['min_eq']:.0f} yrs {' '.join(f'{k}:{v:+.0%}' for k, v in y['yrs'].items())} replay {y['replay_match']}"
         f" | LAG1 SR {l['sr']:.2f} $200->{l['final']:.0f} MDD {l['mdd']:.0%} | score {out['score']:.2f} elig {out['eligible']} ({time.time()-t0:.0f}s)")
    print(s, flush=True); lines.append(s)
    with open("s3_out.txt", "w") as fh:
        fh.write("\n".join(lines))
    json.dump(rows, open("s3_results.json", "w"), indent=1, default=str)
