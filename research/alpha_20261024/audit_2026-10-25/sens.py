from indep_sim import *
import json, sys
FWD=dict(start="2025-01-01",venue="hype")
base,cb,_=run(**FWD)
top=cb.sort_values(ascending=False)
print("TOP 10 contributors fwd (USD):", top.head(10).round(1).to_dict())
print("BOTTOM 5:", top.tail(5).round(1).to_dict(), "total", round(cb.sum(),1))
res={}
for name,kw in [("base",{}),("lag1",dict(lag=1)),("lag2",dict(lag=2)),("cost2x",dict(fee_mult=2)),("cost3x",dict(fee_mult=3)),
                ("no_funding",dict(fund_on=False)),("no_stop",dict(stop=None)),("no_momskip",dict(momskip=1.0)),("no_overlays",dict(stop=None,momskip=1.0)),
                ("no_next_filter",dict(need_next=False)),("rank_within_hype",dict(rank_within_hype=True)),("top30",dict(top_k=30)),("top80",dict(top_k=80)),
                ("tercile",dict(k_mode="ter")),
                ("ex_top5",dict(exclude=tuple(top.index[:5]))),("ex_top10",dict(exclude=tuple(top.index[:10]))),
                ("L7",dict(L=7)),("L21",dict(L=21)),("L30",dict(L=30))]:
    d,c,x=run(**{**FWD,**kw}); s=stats(d); res[name]=s; print(f"{name:18s}",s,flush=True)
json.dump(res,open("sens_fwd.json","w"),indent=1,default=float)
