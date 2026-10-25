"""Test 3 redo: fair permutation null. Each null run keeps the real signal panel but reassigns it to a
randomly permuted set of coin columns (same permutation for all dates): identical turnover and
persistence, only the link signal->coin is broken. 300 runs per strategy/period."""
import os, sys, json
os.environ["ALPHA_FORWARD"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "final"))
import numpy as np, pandas as pd, lab, strategies as S
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SURF, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
COL = {"FCL": "#1baf7a", "CHOP-30": "#2a78d6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF})
U = lab.hype_universe("1d", top_k=50)
P = {f: lab.load(f, "1d") for f in ["open", "high", "low", "close", "qv", "fund"]}
scores = {"FCL": (-P["fund"].rolling(14, min_periods=14).sum(), 1 / 3), "CHOP-30": (-S.chop_ratio(P), 0.2)}
res = {}; nulls = {}
for name, (sc, q) in scores.items():
    for p, a, b in [("IS", "2020-03-01", "2025-01-01"), ("FWD", "2025-01-01", None)]:
        act = lab.backtest(lab.quantile_ls_weights(sc, U, q=q), "1d", cost=lab.HYPE_COST, start=a, end=b)["metrics"]["sharpe"]
        rng = np.random.default_rng(11); null = []
        for j in range(300):
            perm = rng.permutation(sc.shape[1])
            scp = pd.DataFrame(sc.values[:, perm], index=sc.index, columns=sc.columns)
            null.append(lab.backtest(lab.quantile_ls_weights(scp, U, q=q), "1d", cost=lab.HYPE_COST, start=a, end=b)["metrics"]["sharpe"])
        null = np.array(null); nulls[(p, name)] = null
        res[f"{p}|{name}"] = {"actual": float(act), "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)), "p_value": float((null >= act).mean())}
        print(p, name, res[f"{p}|{name}"], flush=True)
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, ((p, name), null) in zip(axs, nulls.items()):
    v = res[f"{p}|{name}"]
    ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.7)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.hist(null, bins=40, color=AXIS, edgecolor=SURF, linewidth=0.4, label="random coin-label books")
    ax.axvline(v["actual"], color=COL[name], lw=2.5, label="actual signal")
    ax.set_title(f"{name} — {'2020-24' if p == 'IS' else '2025-26'}", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(0, 1.0, f"actual {v['actual']:.2f} vs null mean {v['null_mean']:.2f}; p = {v['p_value']:.3f}", transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    ax.set_xlabel("annual Sharpe (ideal book, Binance data, HYPE costs)")
axs[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Test 3: permutation test, H0 = the signal picks coins no better than a random relabelling (300 runs, same turnover & costs)", x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "report_assets", "test3_permutation.png"), dpi=140); plt.close(fig)
R = json.load(open(os.path.join(HERE, "analysis_results.json"))); R["permutation_fair"] = res
json.dump(R, open(os.path.join(HERE, "analysis_results.json"), "w"), indent=1, default=str)
