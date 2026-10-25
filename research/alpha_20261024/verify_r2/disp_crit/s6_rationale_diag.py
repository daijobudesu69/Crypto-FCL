"""Step 6 diagnostics for the economic-rationale critique: what does high disp7 actually measure?
- corr of the disp7 percentile with the trailing alt-vs-BTC spread return (is it disguised spread momentum?)
- upside vs downside composition of dispersion (right tail = pumps)
- link to universe growth (new listings entering the top-50)
- per-year time-in-state and per-state returns (does the edge fade into 2024?)"""
import numpy as np, pandas as pd
import dc
C = dc.C; R = dc.R; BTC = dc.BTC
U = dc.U50()
x7 = np.log(C / C.shift(7)).where(U)
D = x7.std(axis=1); p = dc.trail_pct(D); st = dc.terc_state(p)
alts = U.copy(); alts[BTC] = False
Wa = alts.astype(float).div(alts.sum(1).replace(0, np.nan), axis=0).fillna(0.0)
ralt = (Wa.shift(1) * R).sum(axis=1); spr = np.log1p(ralt) - np.log1p(R[BTC])
spr7 = spr.rolling(7).sum(); spr30 = spr.rolling(30).sum()
med = x7.median(axis=1)
up = x7.quantile(0.9, axis=1) - med; dn = med - x7.quantile(0.1, axis=1)
nnew = (U & (dc.HIST < 180)).sum(axis=1); n = U.sum(axis=1)
ok = p.notna()
L = ["## Step 6 — diagnostics for the rationale critique"]
L.append(f"corr(disp7 percentile, trailing 7d alt-BTC spread) {p[ok].corr(spr7[ok]):+.2f}; with trailing 30d spread {p[ok].corr(spr30[ok]):+.2f}; "
         f"with 7d BTC return {p[ok].corr(np.log(C[BTC] / C[BTC].shift(7))[ok]):+.2f}; with xs median 7d return {p[ok].corr(med[ok]):+.2f}.")
L.append(f"Upside vs downside dispersion: corr(p, 90th-median) {p[ok].corr(up[ok]):+.2f}, corr(p, median-10th) {p[ok].corr(dn[ok]):+.2f}; "
         f"mean (90th-median)/(median-10th) when LONG-SPREAD {(up / dn)[st == 1].mean():.2f} vs SHORT-SPREAD {(up / dn)[st == -1].mean():.2f}.")
L.append(f"Universe growth: corr(p, # members in first 180d) {p[ok].corr(nnew[ok]):+.2f}; mean young members LONG-SPREAD {nnew[st == 1].mean():.1f} vs SHORT-SPREAD {nnew[st == -1].mean():.1f}; "
         f"corr(p, universe size) {p[ok].corr(n[ok]):+.2f}.")
d = pd.read_csv(dc.os.path.join(dc.HERE, "disp7_terc_daily.csv"), index_col=0, parse_dates=True)
r = d["r"]; sa = d["state"]
L += ["", "| year | % days LONG-SPREAD | LONG-SPREAD ann ret (SR) | SHORT-SPREAD ann ret (SR) | alt EW - BTC (log, yr) |", "|---|---|---|---|---|"]
sprA = spr.copy(); sprA.index = sprA.index  # return dated at bar open; approximate
for y in sorted(set(r.index.year)):
    m = r.index.year == y
    a = r[m & (sa == 1)]; b = r[m & (sa == -1)]
    L.append(f"| {y} | {(sa[m] == 1).mean():.0%} | {a.mean() * 365:+.0%} ({dc.sharpe(a):.2f}) | {b.mean() * 365:+.0%} ({dc.sharpe(b):.2f}) | {spr[spr.index.year == y].sum():+.2f} |")
# mean forward 7d spread by state, by year (non-overlapping weekly sampling)
f7 = spr.shift(-1).rolling(7).sum().shift(-6)
wk = f7[st.notna()].iloc[::7]; sw = st.reindex(wk.index)
L.append("Mean fwd 7d alt-BTC log spread (non-overlapping weekly samples), state +1 minus state -1, by year: " +
         ", ".join(f"{y} {wk[(wk.index.year == y) & (sw == 1)].mean() - wk[(wk.index.year == y) & (sw == -1)].mean():+.2%}" for y in sorted(set(wk.index.year))))
dc.append("\n".join(L))
