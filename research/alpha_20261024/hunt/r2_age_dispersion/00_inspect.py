"""Inspect age-bucket composition of the HYPE top-50 universe (no returns computed)."""
import numpy as np, pandas as pd
import r2lib, ages
lab = r2lib.lab
missing = [s for s in ages.OLD_SYMS if s not in lab.load("close", "1d").columns]
print("old-token symbols not in data:", missing)
for kind in ("raw", "clean"):
    A = ages.age_panel(kind)
    for k in (30, 50, 75):
        U = lab.hype_universe("1d", min_hist_days=30, top_k=k)
        print(kind, "top", k)
        for y in [2020, 2021, 2022, 2023, 2024]:
            idx = A.index[A.index.year == y]
            a = A.loc[idx].where(U.loc[idx])
            n = U.loc[idx].sum(1).mean()
            b = [((a >= lo) & (a < hi)).sum(1).mean() for lo, hi in ((30, 180), (180, 365), (365, 730), (730, 1e9))]
            print(f"  {y} univ {n:5.1f} | 30-180 {b[0]:4.1f} 180-365 {b[1]:4.1f} 365-730 {b[2]:4.1f} mature {b[3]:4.1f}")
