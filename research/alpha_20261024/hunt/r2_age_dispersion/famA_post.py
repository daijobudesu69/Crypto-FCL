"""FAMILY A follow-up (after the pre-registered grid came back uniformly NEGATIVE: young coins
outperformed mature coins in every bucket).

Post-hoc, counted (+4 -> family total 42):
 A6 post-cliff bucket: short 365 <= age < 730 vs long MAT (age >= 730), age in {raw, clean}       = 2
 A2 rebalance-1 check of the family-best config                                                    = 1
 16-position version of the family-best (8 most-liquid young short / 8 most-liquid mature long)    = 1
Diagnostics (not trading configs): per-coin PnL contribution of the young (short) leg, the flipped
sign (long young) numbers, top_k 30/75, full stress of the family-best 'none' config.
"""
import numpy as np, pandas as pd, json, warnings
import r2lib, ages
from famA_age import AGE, univ, bucket_weights, R, QV
warnings.filterwarnings("ignore")
lab = r2lib.lab
N_TRIALS = 42
U = univ(50)


def top_liquid(mask, n):
    dqv = QV.rolling(30, min_periods=15).median()
    rk = dqv.where(mask).rank(axis=1, ascending=False)
    return mask & (rk <= n)


def bucket16(A, U, lo, hi, long_side="REST", n=8):
    young = U & (A >= lo) & (A < hi)
    old = U & ((A >= 730) if long_side == "MAT" else (A >= hi))
    young = top_liquid(young, n); old = top_liquid(old, n)
    ny = young.sum(1); no = old.sum(1); ok = (ny > 0) & (no > 0)
    W = (old.astype(float).div(no.replace(0, np.nan), axis=0) - young.astype(float).div(ny.replace(0, np.nan), axis=0))
    return W.mul(ok.astype(float), axis=0).fillna(0.0) / 2


out = {}
for kind in ("raw", "clean"):
    o, _ = r2lib.evaluate(bucket_weights(AGE[kind], U, 365, 730, "MAT"), f"A6_{kind}_365-730_MAT", U=U,
                          rebalance_every=7, full=False)
    r2lib.log(o); print(r2lib.fmt(o, keys=("sharpe", "sh_2020-21", "sh_2022", "sh_2023-24", "avg_n_pos")))

# per-coin contribution of the young leg in the pre-registered core config (raw 30-365 MAT)
W = bucket_weights(AGE["raw"], U, 30, 365, "MAT")
Ws = W.clip(upper=0)
Rn = R.shift(-1).fillna(0.0)
contrib = (Ws * Rn).sum().sort_values()
print("\nYoung (short) leg: total simple PnL contribution by coin (negative = short lost money), worst 10:")
print(contrib.head(10).round(3).to_string())
print("best 5:"); print(contrib.tail(5).round(3).to_string())
print(f"sum all {contrib.sum():.3f}; sum excluding worst 5 {contrib.iloc[5:].sum():.3f}")
out["young_leg_worst10"] = contrib.head(10).round(3).to_dict()
out["young_leg_sum"] = float(contrib.sum()); out["young_leg_sum_ex_worst5"] = float(contrib.iloc[5:].sum())

# flipped sign (long young / short mature) - survivorship artefact, shown for transparency
o, _ = r2lib.evaluate(-W, "FLIP_long_young_raw_30-365_MAT", U=U, rebalance_every=7, full=True, n_trials=N_TRIALS)
print("\n" + r2lib.fmt(o)); out["flip"] = o

# family-best 'none' (highest Sharpe among A1-A4): A2_clean_60-270_REST
best = "A2_clean_60-270_REST"
Wb = bucket_weights(AGE["clean"], U, 60, 270, "REST")
o, _ = r2lib.evaluate(Wb, best, U=U, rebalance_every=7, full=True, n_trials=N_TRIALS)
print("\nBEST " + r2lib.fmt(o)); print("  yearly", o["yearly"]); out["best"] = o
for k in (30, 75):
    Uk = univ(k)
    ok, _ = r2lib.evaluate(bucket_weights(AGE["clean"], Uk, 60, 270, "REST"), f"{best}_top{k}", U=Uk, rebalance_every=7, full=False)
    print("  " + r2lib.fmt(ok, keys=("sharpe", "sh_2020-21", "sh_2022", "sh_2023-24", "avg_n_pos"))); out[f"best_top{k}"] = ok
o1, _ = r2lib.evaluate(Wb, best + "_rb1", U=U, rebalance_every=1, full=False)
r2lib.log(o1); print("  " + r2lib.fmt(o1, keys=("sharpe", "sh_2020-21", "sh_2022", "sh_2023-24")))
W16 = bucket16(AGE["clean"], U, 60, 270, "REST", 8)
o16, _ = r2lib.evaluate(W16, best + "_16pos", U=U, rebalance_every=7, full=True)
r2lib.log(o16); print("  " + r2lib.fmt(o16)); out["best_16pos"] = o16
# neighbours of the best in the pre-registered grid: same kind/long side, lo/hi adjacent
g = pd.read_csv(r2lib.os.path.join(r2lib.HERE, "famA_grid.csv")).set_index("name")["sharpe"]
nb = ["A2_clean_60-180_REST", "A2_clean_60-365_REST", "A2_clean_30-270_REST", "A2_clean_60-270_MAT", "A2_raw_60-270_REST"]
out["best_neighbour_median"] = float(g[nb].median()); print("  neighbour median Sharpe", round(g[nb].median(), 2))
json.dump(out, open(r2lib.os.path.join(r2lib.HERE, "famA_post.json"), "w"), indent=1, default=str)
