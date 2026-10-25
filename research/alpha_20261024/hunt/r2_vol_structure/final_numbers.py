"""Final numbers for the frozen candidates (no new selection). Saves IS daily returns for the lead.
 CAND_B  = B_pkcc30_FLIP_R1  (pre-registered grid cell)  + its 16-pos (8/8) version
 CAND_A  = A_bminus_90_FLIP_BN (pre-registered grid cell, = BAB in disguise) + 16-pos version + top30/75
Trial counts: family A 32 grid + 4 diag = 36; family B 32 grid + 17 robustness = 49; hunter total 85.
"""
import os, json
import numpy as np, pandas as pd
from common import HERE, universe, ls_weights, bn_weights, evaluate, lab, COST
from signals import ohlc, beta_family

TR_A, TR_B, TR_ALL = 36, 49, 85


def dsr_set(o):
    return {f"dsr_n{n}": lab.deflated_sharpe(o["sharpe"], n, o["days"], o["skew"], o["kurt"] + 3)
            for n in (TR_A if o["name"].startswith("A") else TR_B, TR_ALL)}


if __name__ == "__main__":
    out = {}
    U = universe(50)
    o_, h, l, c = ohlc()
    pc = c.shift(1)
    lrange = np.log((h / l).clip(lower=1.0 + 1e-5))
    pk2 = lrange ** 2 / (4 * np.log(2)); cc2 = np.log(c / pc) ** 2
    score = -np.sqrt(pk2.rolling(30, min_periods=24).mean() / cc2.rolling(30, min_periods=24).mean())
    rets = {}
    for nm, W, mask in (("B_pkcc30_FLIP_R1", ls_weights(score, U, q=0.2), U),
                        ("B_pkcc30_FLIP_R1_16pos", ls_weights(score, U, n=8), U)):
        o, res = evaluate(W, nm, mask=mask)
        o.update(dsr_set(o)); out[nm] = o; rets[nm] = res["df"]["r"]
    UA = universe(50, ex_btc=True)
    S = beta_family(90)
    for nm, W, mask, reb in (("A_bminus_90_FLIP_BN", bn_weights(-S["bminus"], UA, S["beta"]), UA, 7),
                             ("A_bminus_90_FLIP_BN_16pos", bn_weights(-S["bminus"], UA, S["beta"], n=8), UA, 7),
                             ("A_bminus_90_FLIP_BN_top30", bn_weights(-S["bminus"], universe(30, True), S["beta"]),
                              universe(30, True), 7),
                             ("A_bminus_90_FLIP_BN_top75", bn_weights(-S["bminus"], universe(75, True), S["beta"]),
                              universe(75, True), 7)):
        o, res = evaluate(W, nm, rebalance=reb, mask=mask)
        o.update(dsr_set(o)); out[nm] = o; rets[nm] = res["df"]["r"]
    df = pd.DataFrame(out).T
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(df.T.to_string())
    pd.DataFrame(rets).to_parquet(os.path.join(HERE, "candidates_daily_IS.parquet"))
    with open(os.path.join(HERE, "final_numbers.json"), "w") as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv) for kk, vv in v.items()}
                   for k, v in out.items()}, f, indent=1, default=str)
