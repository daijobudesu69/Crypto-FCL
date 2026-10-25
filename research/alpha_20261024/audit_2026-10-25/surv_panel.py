import os, pandas as pd, numpy as np
ix=pd.date_range("2024-06-01","2026-07-31",freq="1D",tz="UTC")
C={};Q={};F={};HH={}
hix=pd.date_range("2024-06-01","2026-07-31 23:00",freq="1h",tz="UTC")
for fn in sorted(os.listdir("surv")):
    s=fn[:-4]
    if os.path.getsize(f"surv/{fn}")==0: continue
    d=pd.read_pickle(f"surv/{fn}"); h=d["h"]; f=d["f"]
    h=h.reindex(hix)
    # zero-volume hours = not traded (post-delisting flat prints) -> NaN
    hc=h["close"].where(h["qv"]>0)
    day=h.groupby(h.index.floor("1D"))
    qv=day["qv"].sum(min_count=1)
    nh=hc.groupby(hc.index.floor("1D")).count()
    cl=hc.groupby(hc.index.floor("1D")).last()
    cl=cl.where(nh>=19)  # need >=80% hours traded
    C[s]=cl.reindex(ix); Q[s]=qv.where(nh>=19).reindex(ix); HH[s]=hc.astype("float32")
    if len(f): F[s]=f.groupby((f.index-pd.Timedelta("1s")).floor("1D")).sum().reindex(ix)
P={"C":pd.DataFrame(C),"Q":pd.DataFrame(Q),"F":pd.DataFrame(F),"H":pd.DataFrame(HH)}
pd.to_pickle(P,"surv_panels.pkl")
c=P["C"]; last=c.apply(lambda x:x.last_valid_index())
dead=last[last<ix[-1]-pd.Timedelta("3D")]
print("symbols",c.shape[1],"stopped trading before end (delisted):",len(dead))
