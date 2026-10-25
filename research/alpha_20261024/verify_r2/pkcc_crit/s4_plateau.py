"""Step 4 - plateau. PRE-REGISTERED (written before running), 20 cells, HYPE top-50 unless stated, HYPE cost, daily rebal:
  S1  window W in {20, 25, 30, 40, 45} x quantile q in {0.15, 0.20, 0.30}                     15
  S2  universe top_k in {40, 60}  (W30, q0.2)                                                  2
  S3  estimator: Garman-Klass variance / CC variance (W30, q0.2)                               1
  S4  EWMA ratio, halflife 15d, min 24 obs (q0.2)                                              1
  S5  rank-proportional $-neutral weights (lab.xs_demean_weights on XS rank of -ratio, W30)    1
  -> report median / IQR / share >= 1.0 / share >= 0.6 of Sharpe, lag1 and 2x cost. Report the median, not the max.
  Secondary (5 cells, survivorship-crossed): W in {20,25,30,40,45}, q0.2, min_hist_days=180.
"""
import numpy as np, pandas as pd
from vlib import *

o = lab.load("open", "1d").astype("float64")
h, l, c = hlc()
pc = c.shift(1)
lr = np.log((h / l).clip(lower=1.0 + 1e-5))
pk2 = lr ** 2 / (4 * np.log(2))
cc2 = np.log(c / pc) ** 2
gk2 = (0.5 * lr ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2).clip(lower=0)
U50 = lab.hype_universe("1d", top_k=50)

rows = []
def run(name, W):
    res = bt(W); m = res["metrics"]
    rows.append({"cell": name, "sharpe": m["sharpe"], "lag1": bt(W, lag=1)["metrics"]["sharpe"],
                 "cost2x": bt(W, cost=2 * COST)["metrics"]["sharpe"], **subs(res["df"]["r"])})
    print(rows[-1], flush=True)

for Wn in (20, 25, 30, 40, 45):
    s = -pkcc(Wn, pk2=pk2, cc2=cc2)
    for q in (0.15, 0.20, 0.30):
        run(f"W{Wn}_q{q:.2f}", qls(s, U50, q=q))
s30 = -pkcc(30, pk2=pk2, cc2=cc2)
for k in (40, 60):
    run(f"W30_q0.20_top{k}", qls(s30, lab.hype_universe("1d", top_k=k)))
gk = -np.sqrt(gk2.rolling(30, min_periods=24).mean() / cc2.rolling(30, min_periods=24).mean())
run("GK30_q0.20", qls(gk, U50))
ew = -np.sqrt(pk2.ewm(halflife=15, min_periods=24).mean() / cc2.ewm(halflife=15, min_periods=24).mean())
run("EWMA_hl15_q0.20", qls(ew, U50))
run("W30_rankprop", lab.xs_demean_weights(lab.xs_rank(s30, U50), U50))
main = pd.DataFrame(rows)
rows2 = []
U180 = lab.hype_universe("1d", top_k=50, min_hist_days=180)
for Wn in (20, 25, 30, 40, 45):
    s = -pkcc(Wn, pk2=pk2, cc2=cc2)
    W = qls(s, U180)
    rows2.append({"cell": f"W{Wn}_q0.20_hist180", "sharpe": bt(W)["metrics"]["sharpe"], "lag1": bt(W, lag=1)["metrics"]["sharpe"]})
sec = pd.DataFrame(rows2)
main.to_csv(os.path.join(HERE, "plateau.csv"), index=False)

def summ(x):
    return f"median {x.median():.2f}, IQR [{x.quantile(.25):.2f}, {x.quantile(.75):.2f}], min {x.min():.2f}, max {x.max():.2f}, share>=1.0 {np.mean(x >= 1.0):.0%}, share>=0.6 {np.mean(x >= 0.6):.0%}"

L = ["## Step 4 - plateau (20 pre-registered cells, see s4_plateau.py header)", "",
     "| cell | Sharpe | lag1 | 2x cost | 2020-21 | 2022 | 2023-24 |", "|---|---|---|---|---|---|---|"]
for _, x in main.iterrows():
    L.append(f"| {x['cell']} | {x['sharpe']:.2f} | {x['lag1']:.2f} | {x['cost2x']:.2f} | {x['2020-21']:.2f} | {x['2022']:.2f} | {x['2023-24']:.2f} |")
L += ["",
      f"- Sharpe: {summ(main['sharpe'])}",
      f"- lag1: {summ(main['lag1'])}",
      f"- 2x cost: {summ(main['cost2x'])}",
      f"- min sub-period Sharpe per cell: median {main[['2020-21', '2022', '2023-24']].min(axis=1).median():.2f}; "
      f"cells with every sub-period > 0: {int((main[['2020-21', '2022', '2023-24']].min(axis=1) > 0).sum())}/20",
      "- Secondary, survivorship-crossed (min_hist 180d, q0.2): " + ", ".join(f"{a} {b:.2f}/lag1 {c_:.2f}" for a, b, c_ in sec.values)
      + f" -> median {sec['sharpe'].median():.2f}, lag1 median {sec['lag1'].median():.2f}"]
append("\n".join(L))
print("\n".join(L))
