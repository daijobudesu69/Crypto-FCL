"""Step 4: plateau around the claimed config. PRE-REGISTERED (written before running), 19 cells, all hysteresis
(TERC-style) states, traded universe = signal universe = lab.hype_universe(top_k), cost HYPE_COST:
 A. dispersion lookback L in {5, 7, 10, 14} x percentile window in {180, 365, 730}  (12, incl. baseline L7/365)
 B. band hi/lo in {0.60/0.40, 0.75/0.25, 0.80/0.20} at L7/365                         (3)
 C. top_k in {30, 75} at baseline                                                     (2)
 D. dispersion estimator in {IQR/1.349, MAD*1.4826} at baseline                       (2)
Report the MEDIAN (not the max) of Sharpe, sub-periods and 2024 across the 19 cells."""
import numpy as np, pandas as pd
import dc
U = dc.U50()
cells = []
for L_ in (5, 7, 10, 14):
    for win in (180, 365, 730):
        cells.append((f"L{L_}_w{win}", dict(L=L_, win=win), 50))
for hi, lo in ((0.60, 0.40), (0.75, 0.25), (0.80, 0.20)):
    cells.append((f"band{hi:.2f}/{lo:.2f}", dict(hi=hi, lo=lo), 50))
for k in (30, 75):
    cells.append((f"top{k}", {}, k))
for kind in ("iqr", "mad"):
    cells.append((f"est_{kind}", dict(kind=kind), 50))
assert len(cells) == 19
rows = []
for nm, kw, k in cells:
    Uk = U if k == 50 else dc.U50(k)
    st = dc.disp7_state(Uk, **kw)
    W = dc.spread_w(st, Uk)
    r = dc.bt(W)["df"]["r"]
    r2 = dc.bt(W, cost=2 * dc.COST)["df"]["r"]
    sb = dc.subs(r)
    sw = int((st.dropna().diff().abs() > 0).sum())
    rows.append(dict(cell=nm, sharpe=dc.sharpe(r), s2021=sb["20-21"], s22=sb["22"], s2324=sb["23-24"],
                     s2024=dc.sharpe(r[r.index.year == 2024]), cost2x=dc.sharpe(r2), switches=sw))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows).set_index("cell")
df.round(3).to_csv(dc.os.path.join(dc.HERE, "plateau.csv"))
med = df.median(numeric_only=True)
L = ["## Step 4 — plateau (19 pre-registered cells, written in s4_plateau.py header before running)", "",
     "| cell | Sharpe | 20-21 | 22 | 23-24 | 2024 | 2x cost | switches |", "|---|---|---|---|---|---|---|---|"]
for i, a in df.iterrows():
    L.append(f"| {i} | {a.sharpe:.2f} | {a.s2021:.2f} | {a.s22:.2f} | {a.s2324:.2f} | {a.s2024:.2f} | {a.cost2x:.2f} | {a.switches:.0f} |")
L.append(f"| **MEDIAN** | **{med.sharpe:.2f}** | {med.s2021:.2f} | {med.s22:.2f} | {med.s2324:.2f} | {med.s2024:.2f} | {med.cost2x:.2f} | {med.switches:.0f} |")
L.append(f"Spread: min {df.sharpe.min():.2f}, IQR [{df.sharpe.quantile(.25):.2f}, {df.sharpe.quantile(.75):.2f}], max {df.sharpe.max():.2f}; "
         f"cells > 1.0: {(df.sharpe > 1).sum()}/19, > 0.5: {(df.sharpe > 0.5).sum()}/19; cells with all 3 sub-periods > 0: {((df[['s2021', 's22', 's2324']] > 0).all(1)).sum()}/19; "
         f"cells with 2024 > 0: {(df.s2024 > 0).sum()}/19. Baseline rank among 19: {int((df.sharpe > df.loc['L7_w365', 'sharpe']).sum()) + 1}.")
dc.append("\n".join(L))
