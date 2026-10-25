import sys, pandas as pd, indep_sim as S
h=sys.argv[1]; S.P=pd.read_pickle(f"indep_panels_h{h}.pkl")
for nm,kw in [("fwd",dict(start="2025-01-01",venue="binance")),("is",dict(start="2020-03-01",end="2024-12-31",venue="binance"))]:
    d,_,_=S.run(partial_nan=False,**kw); print("h",h,nm,S.stats(d),flush=True)
