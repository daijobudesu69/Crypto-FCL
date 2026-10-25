from indep_sim import *
import json
res={}
for name,kw in [("fwd_hype",dict(start="2025-01-01",venue="hype")),("fwd_bin",dict(start="2025-01-01",venue="binance")),
                ("is_bin",dict(start="2020-03-01",end="2024-12-31",venue="binance"))]:
    d,c,x=run(**kw); s=stats(d); s.update({k:round(float(v),2) for k,v in x.items()}); res[name]=s; print(name,s,flush=True)
    d.to_csv(f"indep_{name}.csv"); c.sort_values().to_csv(f"indep_contrib_{name}.csv")
