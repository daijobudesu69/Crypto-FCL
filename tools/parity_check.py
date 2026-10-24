"""Hanya di PC dengan data lake: pilihan long/short bot (fclbot.strategy) vs kode riset
(alpha_20261024/fcl.py) di tanggal-tanggal acak 2025-2026, ekuitas $200 (k = 8).

    python tools/parity_check.py
"""
from __future__ import annotations

import os
import sys

os.environ["ALPHA_FORWARD"] = "1"
LAB = r"C:\Crypto data\backtest data and more\alpha_20261024"
sys.path.insert(0, LAB)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fcl  # noqa: E402
import lab  # noqa: E402
from fclbot import config, strategy  # noqa: E402


def research_pick(C, F14, U, mom, t, K):
    ok = U.iloc[t].values & ~np.isnan(F14.iloc[t].values) & ~np.isnan(C.iloc[t].values) & ~np.isnan(C.iloc[t + 1].values)
    idx = np.flatnonzero(ok)
    f, mo = F14.values, mom.values
    mrank = pd.Series(mo[t, idx]).rank(pct=True).values
    cand_s = idx[(mrank <= fcl.MOM_SKIP) | np.isnan(mrank)]
    cand_s = cand_s[np.argsort(-f[t, cand_s], kind="stable")]
    short = list(cand_s[:K])
    cand_l = idx[np.argsort(f[t, idx], kind="stable")]
    long_ = [j for j in cand_l if j not in set(short)][:K]
    cols = C.columns
    return {cols[j] for j in long_}, {cols[j] for j in short}


def main():
    cfg = config.load()
    b2h = {v: k for k, v in cfg.coins.items()}
    C, F, U, F14, mom = fcl.panels()
    qv = lab.load("qv", "1d")
    rng = np.random.default_rng(0)
    days = sorted(rng.choice(np.flatnonzero(C.index >= pd.Timestamp("2025-01-01", tz="UTC"))[:-2], 25, replace=False))
    same_l = same_s = tot = 0
    for t in days:
        cs, f14 = {}, {}
        for sym in C.columns:
            if sym not in b2h:
                continue
            c = C[sym].iloc[: t + 1]
            v = qv[sym].iloc[: t + 1] / c
            df = pd.DataFrame({"close": c.values, "volume": v.values}).dropna()
            cs[b2h[sym]] = df
            if not np.isnan(F14[sym].iloc[t]):
                f14[b2h[sym]] = (float(F14[sym].iloc[t]), 1.0)
        book = strategy.targets(cs, f14, 200.0, set(), cfg.strategy)
        bl = {cfg.coins[c] for c, w in book.weights.items() if w > 0}
        bs = {cfg.coins[c] for c, w in book.weights.items() if w < 0}
        rl, rs = research_pick(C, F14, U, mom, t, book.k)
        same_l += len(bl & rl); same_s += len(bs & rs); tot += book.k
        if bl != rl or bs != rs:
            print(C.index[t].date(), "beda long:", sorted(bl ^ rl), "short:", sorted(bs ^ rs))
    print(f"kecocokan: long {same_l}/{tot}, short {same_s}/{tot} di {len(days)} hari")


if __name__ == "__main__":
    main()
