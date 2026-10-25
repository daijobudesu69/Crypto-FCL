import os, sys, json, time
os.environ["ALPHA_FORWARD"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np, lab, fcl
t=time.time()
out={}
for name, st, en, kw in [("fwd_hype","2025-01-01",None,dict(venue="hype")),
                         ("fwd_binance","2025-01-01",None,dict(venue="binance")),
                         ("is_cap200","2020-03-01","2024-12-31",dict(venue="binance")),
                         ("is_tercile","2020-03-01","2024-12-31",dict(venue="binance",mode="tercile"))]:
    d,s,tr = fcl.run(st, en, **kw)
    s = {k:(round(v,4) if isinstance(v,float) else v) for k,v in s.items()}
    out[name]=s; d.to_csv(os.path.join(os.path.dirname(__file__), f"repro_{name}.csv"))
    pd.DataFrame(tr, columns=["ts","sym","usd"]).to_csv(os.path.join(os.path.dirname(__file__), f"trades_{name}.csv"), index=False)
    print(name, s, round(time.time()-t), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__),"repro.json"),"w"), indent=1, default=str)
