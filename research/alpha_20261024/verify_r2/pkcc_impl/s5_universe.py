"""Step 5 (spec translation check, NOT a selection step): live universe must be built from HYPE data, i.e. ranked
inside the HYPE-listed list rather than top-50-of-150-Binance then intersect. PRE-REGISTERED: D12 rules with
universe = rank within the 92 HYPE symbols by trailing 30d median qv, top 40 (spec choice, fixed in advance) and
top 50 (sensitivity). Full run + fresh 2023 + lag1."""
import numpy as np, pandas as pd
import s3_design as s3
from common_pk import lab
C = lab.load("close", "1d"); qv = lab.load("qv", "1d")
hs = C.columns.isin(lab.hype_symbols())
elig = C.notna() & (C.notna().cumsum() >= 60) & pd.Series(hs, index=C.columns)
dqv = qv.rolling(30, min_periods=15).median()
for K in (40, 50):
    rk = dqv.where(elig).rank(axis=1, ascending=False)
    U2 = elig & (rk <= K)
    s3.M = U2 & s3.S.notna()
    n = s3.M.sum(axis=1)
    out = []
    for nm, kw in (("full", {}), ("y23", {"start": "2023-01-01"}), ("lag1", {"lag": 1})):
        e, Wd, info = s3.dyn(1.0, "q", 0, 1, **kw)
        st = s3.stats(e)
        out.append(f"{nm} SR {st['sr']:.2f} $200->{st['final']:.0f} MDD {st['mdd']:.0%} yrs " +
                   " ".join(f"{k}:{v:+.0%}" for k, v in st['yrs'].items() if k < 2025))
    by = n[n > 0].groupby(n[n > 0].index.year).mean().round(1).to_dict()
    print(f"HYPE-list top{K}: names/yr {by} | " + " | ".join(out), flush=True)
